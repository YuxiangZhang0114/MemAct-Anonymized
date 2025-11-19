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
import json
import logging
import os
import copy
import random
from datetime import datetime
from typing import Any
from uuid import uuid4
import shortuuid

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("mem_agent")
class MemAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        logger.info("Performing class-level ToolAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_single_turn_tokens = config.actor_rollout_ref.rollout.multi_turn.get('max_single_turn_tokens', 4096)
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        logger.info(f"Initialized tools: {cls.tools}")
        cls.random_no_prune = config.actor_rollout_ref.rollout.get("random_no_prune", False)
        logger.info(f"random_no_prune: {cls.random_no_prune}")
        cls.random_no_prune_prob = config.actor_rollout_ref.rollout.get("random_no_prune_prob", 0.4)
        logger.info(f"random_no_prune_prob: {cls.random_no_prune_prob}")

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
        cls.system_prompt_text = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=False)

    def generate_tool_id(self):
        return f"chatcmpl-tool-{uuid4().hex}"

    def generate_short_uuid(self, length: int = 8) -> str:
        try:
            return shortuuid.ShortUUID().random(length=length)
        except ImportError:
            return str(uuid4()).replace('-', '')[:length]
    
    def get_function_list(self, tool_calls):
        function_names = []
        for tool_call in tool_calls:
            if tool_call.get("type") == "function":
                func_name = tool_call["function"].get("name")
                if func_name:
                    function_names.append(func_name)
        return function_names

    async def get_trajactories(self, message_list, prune_num):
        response_mask = []
        response_ids = []
        i = 2
        prune_tool = 0
        while i < len(message_list):
            msg = message_list[i]
            function_list = []
            if msg["role"] == "assistant":
                if "tool_calls" in msg:
                    function_list = self.get_function_list(msg["tool_calls"])
                assistant_msg = [{"role": "assistant", "content": msg["content"]}]
                assistant_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(assistant_msg, tokenize=True, add_generation_prompt=False)
                )
                assistant_ids = assistant_ids[len(self.system_prompt):]
                assistant_ids = assistant_ids[:len(assistant_ids) - 1]
                response_ids.extend(assistant_ids[3:])
                if prune_tool == prune_num or prune_num == -1:
                    response_mask += [1] * (len(assistant_ids) - 3)
                else:
                    response_mask += [0] * (len(assistant_ids) - 3)
                if function_list and "prune_context" in function_list:
                    prune_tool += 1
            elif msg['role'] == 'tool':
                tool_msg = []
                while i < len(message_list) and message_list[i]['role'] == 'tool':
                    tool_msg.append(message_list[i])
                    i += 1
                tool_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(tool_msg, tokenize=True, add_generation_prompt=True)
                )
                tool_ids = tool_ids[len(self.system_prompt):]
                response_ids.extend(tool_ids)
                response_mask += [0] * (len(tool_ids))
                continue
            i += 1
        return response_ids, response_mask
    
    async def get_prune_num(self, message_list):
        prune_num = 0
        for msg in message_list[2:]:
            if msg.get("role") == 'assistant':
                if "tool_calls" in msg:
                    function_list = self.get_function_list(msg["tool_calls"])
                    if 'prune_context' in function_list:
                        prune_num += 1
        return prune_num

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any], is_validate: bool = False) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        
        local_messages = copy.deepcopy(messages)
        actual_messages = copy.deepcopy(messages)
        if self.random_no_prune and random.random() <= self.random_no_prune_prob and not is_validate:
            assert "# Most important:" in messages[0]["content"], "System prompt is not correct"
            assert "</developer_message>\n\n" in messages[1]["content"], "Developer message is not correct"
            messages[0]["content"] = messages[0]["content"].split("# Most important:\nYou may receive a user message containing a `<developer_message>`.")[0]
            messages[1]["content"] = messages[1]["content"].split("</developer_message>\n\n")[-1]
            
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, tools=[_ for _ in self.tool_schemas if _.get('function', {}).get('name') != 'prune_context'][:1], add_generation_prompt=True, tokenize=True
                ),
            )
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, tools=self.tool_schemas, add_generation_prompt=True, tokenize=True
                ),
            )
        instruction_ids = copy.deepcopy(prompt_ids)
        instruction_ids_len = len(prompt_ids)

        user_turns, assistant_turns = 0, 0
        
        prompt_ids_list = []
        response_mask_list = []
        response_ids_list = []
        
        if 'max_tokens' not in sampling_params:
            sampling_params['max_tokens'] = self.max_single_turn_tokens
        while True:
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            assistant_turns += 1

            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
            
            assistant_response = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(response_ids[:-1], skip_special_tokens=False)
            )
            assistant_msg = {
                "role": "assistant",
                "content": assistant_response
            }
            local_messages.append(assistant_msg.copy())
            actual_messages.append(assistant_msg.copy())
            
            assistant_response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template([assistant_msg], tokenize=True, add_generation_prompt=False)
            )
            assistant_response_ids = assistant_response_ids[len(self.system_prompt) + 3:]
            assistant_response_ids = assistant_response_ids[:len(assistant_response_ids) - 1]
            prompt_ids += assistant_response_ids

            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break
            
            if not tool_calls:
                break

            no_delete_actual_messages = copy.deepcopy(actual_messages)
            tasks = []
            prune_responses = []
            deleted_prompt_ids = []
            prune_flag = 0
            for tool_call in tool_calls[: self.max_parallel_calls]:
                if tool_call.name == 'prune_context':
                    prune_flag = 1
                    prune_tool_id = self.generate_short_uuid()
                    try:
                        tool_args = json.loads(tool_call.arguments)
                        delete_ids = tool_args.get("delete_ids", [])
                        memory = tool_args.get("memory", "")
                        prune_result, deleted_prompt_ids = await self.prune_actual_messages_prompt_ids(actual_messages, delete_ids, memory, instruction_ids, prune_tool_id)

                        prune_responses.append({
                            "role": "tool",
                            "content": prune_result
                        })
                    except Exception as e:
                        logger.exception(f"Error when executing prune_context: {e}")
                        prune_responses.append({
                            "role": "tool",
                            "content": json.dumps({'success': False, 'message': 'Wrong Wrong Wrong.\n', 'tool_call_id': prune_tool_id})
                        })
                else:
                    tasks.append(self._call_tool(tool_call))
            if prune_flag:
                prune_num = await self.get_prune_num(no_delete_actual_messages)
                temp_response_ids, temp_response_mask = await self.get_trajactories(no_delete_actual_messages, prune_num)
                response_ids_list.append(temp_response_ids)
                response_mask_list.append(temp_response_mask)
            
            other_tool_responses = []
            if tasks:
                with simple_timer("tool_calls", metrics):
                    other_tool_responses = await asyncio.gather(*tasks)

            if any(isinstance(item, Exception) for item in other_tool_responses):
                break

            tool_responses = []
            tool_call_ids = [self.generate_tool_id() for _ in tool_calls[: self.max_parallel_calls]]

            local_messages[-1]["tool_calls"] = [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments
                    }
                }
                for tool_call_id, tool_call in zip(tool_call_ids, tool_calls[: self.max_parallel_calls])
            ]
            
            actual_messages[-1]["tool_calls"] = [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments
                    }
                }
                for tool_call_id, tool_call in zip(tool_call_ids, tool_calls[: self.max_parallel_calls])
            ]

            prune_tool_num = 0
            other_tool_num = 0
            for tool_call_id, tool_call in zip(tool_call_ids, tool_calls[: self.max_parallel_calls]):
                if tool_call.name == 'prune_context':
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": prune_responses[prune_tool_num]["content"]
                    }
                    local_messages.append(tool_msg.copy())
                    actual_messages.append(tool_msg.copy())
                    tool_responses.append(prune_responses[prune_tool_num])
                    prune_tool_num += 1
                else:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": other_tool_responses[other_tool_num]["content"]
                    }
                    local_messages.append(tool_msg.copy())
                    actual_messages.append(tool_msg.copy())
                    tool_responses.append(other_tool_responses[other_tool_num])
                    other_tool_num += 1

            tool_response_ids = await self.loop.run_in_executor(
                None,
                lambda messages=tool_responses: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True
                ),
            )
            tool_response_ids = tool_response_ids[len(self.system_prompt):]

            if deleted_prompt_ids:
                prompt_ids = deleted_prompt_ids.copy()

            prompt_ids += tool_response_ids
            
            if len(prompt_ids) - instruction_ids_len >= self.response_length:
                break
            
            user_turns += 1

        assert len(response_ids_list) == len(response_mask_list), "Response IDs, and response mask lists length mismatch"
        temp_response_ids, temp_response_mask = await self.get_trajactories(actual_messages, -1)
        prompt_ids = prompt_ids[: instruction_ids_len]
        
        response_ids_list.append(temp_response_ids)
        response_mask_list.append(temp_response_mask)
        prompt_ids_list = [prompt_ids.copy() for _ in range(len(response_mask_list))]

        output_list = []
        for idx in range(len(prompt_ids_list)):
            output = AgentLoopOutput(
                prompt_ids = prompt_ids_list[idx][-self.prompt_length:],
                response_ids = response_ids_list[idx][: self.response_length],
                response_mask = response_mask_list[idx][: self.response_length],
                num_turns=user_turns + assistant_turns + 1,
                metrics = metrics
            )
            output_list.append(output)
            
        session_uuid = self.generate_short_uuid()
        await self._write_messages_to_log(local_messages, request_id, 'local', session_uuid)
        await self._write_messages_to_log(actual_messages, request_id, 'actual', session_uuid)
        return output_list


    async def _call_tool(self, tool_call: FunctionCall) -> dict[str, str]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]

            instance_id = await tool.create()
            tool_response, _, _ = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e
        finally:
            if tool and instance_id:
                await tool.release(instance_id)
        if tool_response is None:
            tool_response = "Tool execution failed."
        
        if len(tool_response) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response = tool_response[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response = "(truncated)..." + tool_response[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response = tool_response[:length] + "...(truncated)..." + tool_response[-length:]

        return {
            "role": "tool",
            "content": tool_response,
        }

    def restore_as_text(self, tool_call):
        name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]
        return f"<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>\n"
    
    def tool_response_as_text(self, tool_response):
        return f"\n<tool_response>\n{tool_response['content']}\n</tool_response>"
    
    async def prune_actual_messages_prompt_ids(self, actual_messages: list[dict[str, Any]], delete_ids: list[str], memory: str, instruction_ids: list[str], prune_tool_id: str):
        """Prune messages and prompt_ids based on delete_ids and memory."""
        
        if not delete_ids or not memory:
            return json.dumps({'success': False, 'message': 'Context manager failed to execute.\n- No delete IDs or no memory provided\n', 'tool_call_id': prune_tool_id}), []

        tool_call_ids_to_delete = []
        for msg in actual_messages:
            if msg.get("role") == "tool":
                content = str(msg.get("content", ""))
                for uid in delete_ids:
                    if uid and uid in content:
                        tool_call_id = msg.get("tool_call_id")
                        tool_call_ids_to_delete.append(tool_call_id)
                        break

        if not tool_call_ids_to_delete:
            return json.dumps({'success': False, 'message': 'Context manager failed to execute.\n- No tool calls matched for deletion.', 'tool_call_id': prune_tool_id}), []

        messages_to_remove = []

        id = 0
        while id < len(actual_messages):
            msg = actual_messages[id]
            if msg.get("role") == 'assistant':
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    if all(tc.get("id") in tool_call_ids_to_delete for tc in tool_calls):
                        messages_to_remove.append(msg)
                    else:
                        remain_tool_call_msg = []
                        for tc in tool_calls:
                            if tc.get("id") not in tool_call_ids_to_delete:
                                remain_tool_call_msg.append(tc)
                            else:
                                msg["content"] = msg["content"].replace(self.restore_as_text(tc), "")
                                
                        if remain_tool_call_msg:
                            msg["tool_calls"] = remain_tool_call_msg

            elif msg.get("role") == 'tool' and (msg.get("tool_call_id") in tool_call_ids_to_delete):
                messages_to_remove.append(msg)

            id += 1

        for msg in messages_to_remove:
            actual_messages.remove(msg)
        
        deleted_prompt_ids = copy.deepcopy(instruction_ids[:len(instruction_ids) - 3])
        i = 2
        while i < len(actual_messages):
            msg = actual_messages[i]
            if msg["role"] == "assistant":
                assistant_msg = [{"role": "assistant", "content": msg["content"]}]
                assistant_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(assistant_msg, tokenize=True, add_generation_prompt=False)
                )
                assistant_ids = assistant_ids[len(self.system_prompt):]
                assistant_ids = assistant_ids[:len(assistant_ids) - 1]
                deleted_prompt_ids.extend(assistant_ids)
            elif msg['role'] == 'tool':
                tool_msg = []
                while i < len(actual_messages) and actual_messages[i]['role'] == 'tool':
                    tool_msg.append(actual_messages[i])
                    i += 1
                tool_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(tool_msg, tokenize=True, add_generation_prompt=False)
                )
                tool_ids = tool_ids[len(self.system_prompt):]
                deleted_prompt_ids.extend(tool_ids)
                continue
            i += 1
        
        prune_result = json.dumps({'success': True, 'message': f'Context manager run successfully. - Successfully deleted {len(tool_call_ids_to_delete)} tool call records and responses', 'tool_call_id': prune_tool_id})
        return prune_result, deleted_prompt_ids

    async def _write_messages_to_log(self, messages: list[dict[str, Any]], request_id: str, log_type: str, session_uuid: str) -> None:
        """Write conversation messages to a local log file."""
        try:
            if log_type == 'local':
                log_dir = os.path.join(os.getcwd(), "logs", "conversation_logs")
            else:
                log_dir = os.path.join(os.getcwd(), "logs", "actual_logs")
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if log_type == 'local':
                filename = f"conversation_{timestamp}_{request_id[:8]}_{session_uuid}.json"
            else:
                filename = f"actual_{timestamp}_{request_id[:8]}_{session_uuid}.json"
            log_file_path = os.path.join(log_dir, filename)
            
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "conversation": messages,
                "total_turns": len([msg for msg in messages if msg["role"] in ["user", "assistant"]]),
                "tool_calls_count": len([msg for msg in messages if msg["role"] == "tool"])
            }
            
            await self.loop.run_in_executor(
                None,
                lambda: self._write_json_file(log_file_path, log_data)
            )
            
            logger.info(f"Conversation logged to: {log_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to write conversation log: {e}")
    
    def _write_json_file(self, file_path: str, data: dict) -> None:
        """Synchronous helper to write JSON data to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    async def _check_output_list_to_log(self, output_list: list[AgentLoopOutput], request_id: str, session_uuid: str) -> None:
        """Write output list details to log files for debugging and analysis."""
        try:
            log_dir = os.path.join(os.getcwd(), "logs", "agent_outputs")
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_output_{timestamp}_{request_id[:8]}_{session_uuid}.json"
            log_file_path = os.path.join(log_dir, filename)
            
            outputs_data = []
            
            for idx, output in enumerate(output_list):
                prompt_ids = output.prompt_ids
                response_ids = output.response_ids
                response_mask = output.response_mask
                
                prompt_text = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                )
                response_text = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(response_ids, skip_special_tokens=False)
                )
                
                token_details = []
                for pos, (token_id, mask) in enumerate(zip(response_ids, response_mask)):
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    token_details.append({
                        "position": pos,
                        "token_id": int(token_id),
                        "token_text": repr(token_text),
                        "mask": int(mask)
                    })
                
                output_data = {
                    "output_num": len(output_list) + 1,
                    "output_index": idx,
                    "prompt_text": prompt_text,
                    "response_text": response_text,
                    "token_details": token_details,
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "response_mask": response_mask,
                    "masked_token_count": sum(response_mask),
                    "total_token_count": len(response_mask)
                }
                outputs_data.append(output_data)
            
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "total_outputs": len(output_list),
                "outputs": outputs_data
            }
            
            await self.loop.run_in_executor(
                None,
                lambda: self._write_json_file(log_file_path, log_data)
            )
            
            logger.info(f"Agent outputs logged to: {log_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to write agent output log: {e}")
