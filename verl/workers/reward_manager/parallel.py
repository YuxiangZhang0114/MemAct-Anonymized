# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("parallel")
class ParallelRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", max_workers=8, timeout=15) -> None:
        """
        Initialize the ParallelRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            max_workers: Max parallel workers for computing scores. Defaults to `min(32, (os.cpu_count() or 1) + 4)` from ThreadPoolExecutor if None.
            timeout: Optional per-item timeout (in seconds). If set, slow computations will be treated as failures with score 0.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.max_workers = max_workers
        self.timeout = timeout

    def _run_async(self, coro):
        """Run a coroutine safely in a fresh event loop.

        This avoids interference with external event loops and makes the manager robust
        against different `compute_score` implementations (sync/async).
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

    def _call_compute_score(self, *, data_source, solution_str, ground_truth, extra_info):
        try:
            fn = self.compute_score
            # If compute_score is declared async
            if asyncio.iscoroutinefunction(fn):
                result = self._run_async(
                    fn(
                        data_source=data_source,
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
                )
            else:
                result = fn(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

            # If a coroutine object is returned (even if function not marked async)
            if asyncio.iscoroutine(result):
                result = self._run_async(result)

            return {"ok": True, "result": result}
        except Exception as e:  # Robust against any compute_score failure
            return {
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Prepare tasks (decode on main thread, independent of compute_score implementation)
        tasks = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            non_tensor = data_item.non_tensor_batch
            ground_truth = (
                (non_tensor.get("reward_model") or {}).get("ground_truth")
                if isinstance(non_tensor.get("reward_model"), dict)
                else None
            )
            data_source = non_tensor.get(self.reward_fn_key)
            extra_info = non_tensor.get("extra_info", {}) or {}
            num_turns = non_tensor.get("__num_turns__", None)
            try:
                # Ensure dict-like and not mutate original reference across threads
                extra_info = dict(extra_info)
                extra_info["num_turns"] = num_turns
            except Exception:
                extra_info = {"num_turns": num_turns}

            tasks.append(
                {
                    "i": i,
                    "valid_response_length": valid_response_length,
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "extra_info": extra_info,
                }
            )

        # Launch parallel compute
        futures_map = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for payload in tasks:
                fut = executor.submit(
                    self._call_compute_score,
                    data_source=payload["data_source"],
                    solution_str=payload["response_str"],
                    ground_truth=payload["ground_truth"],
                    extra_info=payload["extra_info"],
                )
                futures_map[fut] = payload

            # Gather results (keep deterministic order by staging first)
            staged_results = {}
            for fut in as_completed(futures_map.keys()):
                payload = futures_map[fut]
                i = payload["i"]
                try:
                    result = fut.result(timeout=self.timeout)
                except Exception as e:
                    # Capture timeout or executor-level errors
                    result = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
                staged_results[i] = {"payload": payload, "compute": result}

        # Aggregate on main thread in sample order for stable behavior
        for i in range(len(tasks)):
            staged = staged_results[i]
            payload = staged["payload"]
            compute = staged["compute"]

            data_source = payload["data_source"]
            prompt_str = payload["prompt_str"]
            response_str = payload["response_str"]
            ground_truth = payload["ground_truth"]
            valid_response_length = payload["valid_response_length"]

            score_obj = None
            if compute.get("ok"):
                score_obj = compute.get("result")
            else:
                # Failure path: supply default reward and attach diagnostics
                score_obj = {
                    "score": 0.0,
                    "error": compute.get("error"),
                    "traceback": compute.get("traceback"),
                }

            if isinstance(score_obj, dict):
                reward = score_obj.get("score", 0.0)
                # Store the information including original reward and any diagnostics
                for key, value in score_obj.items():
                    reward_extra_info[key].append(value)
                printable = score_obj
            else:
                # Try best-effort conversion
                try:
                    reward = float(score_obj)
                except Exception:
                    reward = 0.0
                    reward_extra_info["error"].append("invalid_score_type")
                    reward_extra_info["raw_score"].append(score_obj)
                    printable = {"score": reward, "raw_score": score_obj, "error": "invalid_score_type"}
                else:
                    printable = reward

            try:
                reward_tensor[i, valid_response_length - 1] = reward
            except Exception:
                # Guard against empty responses or invalid lengths
                try:
                    # If valid_response_length is a 0-d tensor, try to convert
                    idx = int(valid_response_length) - 1
                    if idx >= 0 and idx < reward_tensor.shape[1]:
                        reward_tensor[i, idx] = reward
                except Exception:
                    # As a last resort, place reward at the last position
                    reward_tensor[i, -1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(printable, dict):
                    for key, value in printable.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", printable)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
