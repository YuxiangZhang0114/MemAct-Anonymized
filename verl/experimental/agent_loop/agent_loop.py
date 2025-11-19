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
import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any
import uuid
from concurrent.futures import ThreadPoolExecutor

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.async_server import async_server_class
from verl.utils.reward_score.mem_agent import compute_score

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            List[int]: List of generated token ids.
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    uid: str = None
    """prompt_id"""
    rollout_id: str = None
    """rollout_id"""
    reward: float = None
    


# make hydra.utils.instantiate happy
class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(
        self, trainer_config: _DummyConfig, server_manager: AsyncLLMServerManager, tokenizer: AutoTokenizer, **kwargs
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.init_class(trainer_config.config, tokenizer, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, **kwargs):
        """This is used to do heavy initialization work that should shared across all instances. It's only called once.

        Args:
            config (DictConfig): trainer config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            **kwargs: extra kwargs from config file passed in by `hydra.utils.instantiate`.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            messages (List[Dict[str, Any]]): Input messages.
            sampling_params (Dict[str, Any]): LLM sampling params.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


@ray.remote
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle]):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config

        trace_config = config.trainer.get("rollout_trace", {})
        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<-  llm deletecall ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 1, 1, ..., 1        |
            attention_mask: | 1, 1, 1, ..., 1, 1 | 1, 1, .., 1, 1 | 1, 1, 1, ..., 1, 1 | 1, 1, ..., 1        |
            
            responses:     |<- LLM generation ->|<-  llm deletecall ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: |  0, 0, .., 0, 0   | 0, 0, .., 0, 0       | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
            attention_mask: | 1, 1, 1, ..., 1, 1 | 1, 1, .., 1, 1       | 1, 1, .., 1, 1 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
        )
        # print("batch input")
        # print(batch.batch.keys())
        # print(batch.non_tensor_batch)
        # print(batch.meta_info)

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        tasks = []
        agent_names = batch.non_tensor_batch["agent_name"]
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        extra_info = batch.non_tensor_batch['extra_info']
        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(raw_prompts))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        )

        for agent_name, messages, trajectory in zip(agent_names, raw_prompts, trajectory_info, strict=True):
            tasks.append(
                asyncio.create_task(self._run_agent_loop(agent_name, messages.tolist(), sampling_params, trajectory))
            )
        outputs = await asyncio.gather(*tasks)

        # Compute score in parallel using 20 worker threads
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Prepare parameters for computing score
            score_params = []
            for idx, output in enumerate(outputs):
                output_text = self.tokenizer.decode(output[-1].response_ids)
                score_params.append((output_text, extra_info[idx]))
            
            # Execute compute_score in parallel
            output_rewards = list(executor.map(lambda x: compute_score(*x), score_params))
            
            # Assign results to corresponding output
            for idx, output in enumerate(outputs):
                for item in output:
                    item.reward = output_rewards[idx]
                
        #   for item in output:
        #     item.rollout_id = rollout_id
        # [[o1,o2,o2,o3],[],[],[]]
        if batch.meta_info.get("validate", False):
            outputs = [output[-1] for output in outputs]
            # [o-1,o-2,o-3]

        else:
            if config.get("actual_n",None) is not None:
                # if the output is a list, it means the output is a list of lists, we need to flatten it
                rollout_ids = []
                uids = []
                flat_outputs = []
                for i, output in enumerate(outputs):
                    rollout_ids.extend([str(uuid.uuid4())]*len(output))
                    uids.extend([batch.non_tensor_batch["uid"][i]]*len(output))
                    flat_outputs.extend(output)
                
                for rollout_id, flat_output, uid in zip(rollout_ids,flat_outputs,uids):
                    flat_output.rollout_id = rollout_id
                    flat_output.uid = uid

                outputs = flat_outputs
                # [o1,o2,o3,o4,o5,o6,o7,o8,o9,o10]
            
        output = self._postprocess(outputs)
        
        
        return output

    async def _run_agent_loop(
        self,
        agent_name: str,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
    ) -> AgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
            )
            assert "validate" in trajectory, "validate not found in trajectory"
            
            output = await agent_loop.run(messages, sampling_params,is_validate=trajectory.get("validate", False))
            return output

    def _postprocess(self, inputs: list[AgentLoopOutput]) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts
        self.tokenizer.padding_side = "left"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.prompt_ids} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # responses
        self.tokenizer.padding_side = "right"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_ids} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # response_mask
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_mask} for input in inputs],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        response_mask = outputs["input_ids"]
        assert response_ids.shape == response_mask.shape, (
            f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
        )
        response_mask = response_mask * response_attention_mask

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([input.num_turns for input in inputs], dtype=np.int32)
        uids = np.array([input.uid for input in inputs], dtype=object)
        rollout_ids = np.array([input.rollout_id for input in inputs], dtype=object)
        rewards = np.array([input.reward for input in inputs], dtype=np.float32)

        metrics = [input.metrics.model_dump() for input in inputs]
        return DataProto(batch=batch, 
                         non_tensor_batch={"__num_turns__": num_turns, "rollout_ids": rollout_ids,"uids":uids,"agent_reward":rewards}, 
                         meta_info={"metrics": metrics})


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        self.sleep()

    def _initialize_llm_servers(self):
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    logger.warning(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        for i in range(self.config.actor_rollout_ref.rollout.agent.num_workers):
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                ).remote(self.config, self.async_llm_servers)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        validate = prompts.meta_info.get("validate", False)
        output = DataProto.concat(outputs)

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # calculate performance metrics
        metrics = [output.meta_info["metrics"] for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing}
    
        if self.config.actor_rollout_ref.rollout.get("actual_n",None) is not None and not validate:
            
            # Executing DCPO sampling
            
            assert self.config.actor_rollout_ref.rollout.n >= self.config.actor_rollout_ref.rollout.actual_n, "n must be greater than or equal to actual_n"
            
            if self.config.actor_rollout_ref.rollout.get("dcpo_sampling_strategy",'random') == 'random':
                output = self._sampling_random(output)
  
            else:
                raise ValueError(f"Invalid sampling method: {self.config.actor_rollout_ref.rollout.get('sampling_method', 'random')}")
            
        return output

    def _sampling_random(self, output: DataProto):
        """
        DCPO sampling (corrected version):
        - Sample exactly n items for each uid
        - When unique_rollout_ids <= n: each rollout gets at least 1 item, remaining filled with round-robin
        - When unique_rollout_ids > n: select n different rollouts, each gets 1 item
        - Try to avoid duplicate samples; allow sampling with replacement when insufficient
        """
        import random
        from collections import defaultdict
        from tensordict import TensorDict
        import numpy as np

        target_n = self.config.actor_rollout_ref.rollout.n

        # Validate required fields
        if "uids" not in output.non_tensor_batch:
            raise KeyError("uids not found in output.non_tensor_batch")
        if "rollout_ids" not in output.non_tensor_batch:
            raise KeyError("rollout_ids not found in output.non_tensor_batch")

        uids = output.non_tensor_batch["uids"]
        rollout_ids = output.non_tensor_batch["rollout_ids"]

        # Build uid -> (rid -> [indices]) mapping
        per_uid = defaultdict(lambda: defaultdict(list))
        for idx, (u, r) in enumerate(zip(uids, rollout_ids)):
            per_uid[u][r].append(idx)

        selected_indices = []

        for u, rid2idxs in per_uid.items():
            rids = list(rid2idxs.keys())
            K = len(rids)

            if K >= target_n:
                # Can only cover n different rollouts
                chosen_rids = random.sample(rids, k=target_n)
                for rid in chosen_rids:
                    idx = random.choice(rid2idxs[rid])
                    selected_indices.append(idx)
            else:
                # First, take 1 from each rollout
                used_per_rid = {rid: set() for rid in rids}
                for rid in rids:
                    idx = random.choice(rid2idxs[rid])
                    selected_indices.append(idx)
                    used_per_rid[rid].add(idx)

                # Round-robin fill remaining n-K items
                remaining = target_n - K
                rr = rids[:]  # Round-robin list
                random.shuffle(rr)
                pos = 0
                while remaining > 0:
                    rid = rr[pos]
                    # Prefer unused indices
                    candidates = [i for i in rid2idxs[rid] if i not in used_per_rid[rid]]
                    if candidates:
                        idx = random.choice(candidates)
                    else:
                        # Allow sampling with replacement when insufficient
                        idx = random.choice(rid2idxs[rid])
                    selected_indices.append(idx)
                    used_per_rid[rid].add(idx)
                    remaining -= 1
                    pos = (pos + 1) % len(rr)

        # Don't sort, maintain stability after uid aggregation
        # Rebuild tensors/arrays
        new_batch = {k: v[selected_indices] for k, v in output.batch.items()}

        new_non = {}
        for k, arr in output.non_tensor_batch.items():
            new_non[k] = np.asarray(arr)[selected_indices]

        sampled = DataProto(
            batch=TensorDict(new_batch, batch_size=len(selected_indices)),
            non_tensor_batch=new_non,
            meta_info=output.meta_info,
        )

        # Strictly validate return quantity
        expected = len(set(uids)) * target_n
        actual = len(selected_indices)
        assert actual == expected, f"Sampling size mismatch: expected {expected}, got {actual}"

        return sampled        
        
    
    def _sampling_weighted(
        self,
        output: DataProto,
        *,
        target_n: int | None = None,
        weight_key_candidates: tuple[str, ...] = ("dcpo_score", "score", "reward", "quality", "value", "logprob"),
        rid_weight_agg: str = "mean",  # "mean" | "max" | "sum"
        temperature: float = 1.0,
        seed: int | None = None,
    ):
        """
        DCPO weighted sampling (grouped by uid):
        - Sample exactly target_n items for each uid
        - Support sample-level weights and aggregate to rollout-level weights
        - K := unique_rollout_ids
            * If K >= n: select n different rollouts without replacement by rollout weights, each gets 1 item (within sample, use sample weights)
            * If K < n: first ensure each rollout gets at least 1 item; remaining filled with weighted round-robin by rollout weights
        - Weight temperature scaling: p ∝ (w+ε)^(1/temperature)
        """
        import random
        import numpy as np
        from collections import defaultdict
        from tensordict import TensorDict

        rng = random.Random(seed)

        if target_n is None:
            target_n = self.config.actor_rollout_ref.rollout.n

        # Validate required fields
        if "uids" not in output.non_tensor_batch:
            raise KeyError("uids not found in output.non_tensor_batch")
        if "rollout_ids" not in output.non_tensor_batch:
            raise KeyError("rollout_ids not found in output.non_tensor_batch")

        uids = output.non_tensor_batch["uids"]
        rids = output.non_tensor_batch["rollout_ids"]

        # Parse sample-level weights (default to all 1s if not found)
        weight_arr = None
        for k in weight_key_candidates:
            if k in output.non_tensor_batch:
                weight_arr = np.asarray(output.non_tensor_batch[k], dtype=float)
                break
        if weight_arr is None:
            weight_arr = np.ones(len(uids), dtype=float)

        # Clean weights and apply temperature scaling
        weight_arr = np.nan_to_num(weight_arr, nan=0.0, posinf=0.0, neginf=0.0)
        weight_arr = np.maximum(weight_arr, 0.0)
        eps = 1e-12
        t = max(float(temperature), 1e-6)
        sample_w = np.power(weight_arr + eps, 1.0 / t)

        # Build uid -> (rid -> [indices]) mapping
        per_uid = defaultdict(lambda: defaultdict(list))
        for idx, (u, r) in enumerate(zip(uids, rids)):
            per_uid[u][r].append(idx)

        def _normalize(probs: np.ndarray) -> np.ndarray:
            s = float(np.sum(probs))
            if s <= 0 or not np.isfinite(s):
                return np.ones_like(probs) / len(probs)
            return probs / s

        def _choice_wo_replacement(items: list, probs: np.ndarray, k: int) -> list:
            # Weighted sampling without replacement (simple sequential renormalization)
            items = list(items)
            probs = np.array(probs, dtype=float)
            chosen = []
            k = min(k, len(items))
            for _ in range(k):
                p = _normalize(probs)
                r = rng.random()
                c = 0.0
                pick = 0
                for i, pi in enumerate(p):
                    c += pi
                    if r <= c:
                        pick = i
                        break
                chosen.append(items[pick])
                items.pop(pick)
                probs = np.delete(probs, pick)
            return chosen

        selected_indices: list[int] = []

        for u, rid2idxs in per_uid.items():
            rid_list = list(rid2idxs.keys())
            K = len(rid_list)

            # Aggregate rollout-level weights
            rid_weights = []
            for rid in rid_list:
                idxs = rid2idxs[rid]
                ws = sample_w[idxs]
                if rid_weight_agg == "max":
                    rw = float(np.max(ws)) if len(ws) else 0.0
                elif rid_weight_agg == "sum":
                    rw = float(np.sum(ws)) if len(ws) else 0.0
                else:  # mean
                    rw = float(np.mean(ws)) if len(ws) else 0.0
                rid_weights.append(rw)
            rid_weights = np.array(rid_weights, dtype=float)

            if K >= target_n:
                # Select n different rollouts (weighted without replacement)
                chosen_rids = _choice_wo_replacement(rid_list, rid_weights, target_n)
                for rid in chosen_rids:
                    idxs = rid2idxs[rid]
                    ps = _normalize(sample_w[idxs])
                    rnum = rng.random()
                    c = 0.0
                    for j, pj in enumerate(ps):
                        c += pj
                        if rnum <= c:
                            selected_indices.append(idxs[j])
                            break
            else:
                # 1) First ensure each rollout gets at least 1 item (within sample, use weights)
                used_by_rid = {rid: set() for rid in rid_list}
                for rid in rid_list:
                    idxs = rid2idxs[rid]
                    ps = _normalize(sample_w[idxs])
                    rnum = rng.random()
                    c = 0.0
                    for j, pj in enumerate(ps):
                        c += pj
                        if rnum <= c:
                            chosen = idxs[j]
                            selected_indices.append(chosen)
                            used_by_rid[rid].add(chosen)
                            break

                # 2) Remaining n-K items: weighted round-robin fill by rollout weights
                remaining = target_n - K
                if remaining > 0:
                    p_rid = _normalize(rid_weights)
                    while remaining > 0:
                        # Select rollout
                        rnum = rng.random()
                        c = 0.0
                        pick = 0
                        for i, pi in enumerate(p_rid):
                            c += pi
                            if rnum <= c:
                                pick = i
                                break
                        rid = rid_list[pick]
                        idxs = rid2idxs[rid]

                        # Prefer unused samples
                        unused = [i for i in idxs if i not in used_by_rid[rid]]
                        if unused:
                            ps = _normalize(sample_w[unused])
                            r2 = rng.random()
                            c2 = 0.0
                            for j, pj in enumerate(ps):
                                c2 += pj
                                if r2 <= c2:
                                    chosen = unused[j]
                                    break
                        else:
                            # Allow sampling with replacement when insufficient
                            ps = _normalize(sample_w[idxs])
                            r2 = rng.random()
                            c2 = 0.0
                            for j, pj in enumerate(ps):
                                c2 += pj
                                if r2 <= c2:
                                    chosen = idxs[j]
                                    break

                        selected_indices.append(chosen)
                        used_by_rid[rid].add(chosen)
                        remaining -= 1

        # Rebuild DataProto (don't sort, avoid cross-uid shuffling)
        new_batch = {k: v[selected_indices] for k, v in output.batch.items()}

        # Convert non-tensor part to np before slicing, preserve dtype
        new_non = {}
        for k, arr in output.non_tensor_batch.items():
            new_non[k] = np.asarray(arr)[selected_indices]

        sampled = DataProto(
            batch=TensorDict(new_batch, batch_size=len(selected_indices)),
            non_tensor_batch=new_non,
            meta_info=output.meta_info,
        )

        # Strictly validate size
        expected = len(set(uids)) * target_n
        actual = len(selected_indices)
        assert actual == expected, f"Sampling size mismatch: expected {expected}, got {actual}"

        return sampled

    
    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout server instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all rollout server instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])
