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

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.utils.search_r1_like_utils import perform_single_search_batch
from verl.utils.rollout_trace import rollout_trace_op

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

import re

def decode_unicode_escapes_only(s: str) -> str:
    # Convert \uXXXX escape sequences to characters
    s2 = re.sub(r'\\u([0-9a-fA-F]{4})',
                lambda m: chr(int(m.group(1), 16)), s)
    # Merge surrogates (ensure sequences like \uD83D\uDE00 become a single code point)
    try:
        s2 = s2.encode('utf-16-le', 'surrogatepass').decode('utf-16-le')
    except UnicodeError:
        pass
    return s2


class SearchTool(BaseTool):
    """Search tool for retrieving information using external retrieval services.

    This tool provides search functionality for semantic search operations.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for relevant information based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries"
                            }
                        },
                        "required": ["query_list"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Configuration
        self.timeout = config.get("timeout", 30)

        # Retrieval service configuration
        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        self.topk = config.get("topk", 5)
        if self.retrieval_service_url == "":
            raise ValueError("retrieval_service_url is not set")

        logger.info(f"Initialized SearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id

    def execute_search(self, instance_id: str, query_list: list, retrieval_service_url: str, topk: int, timeout: int):
        """Execute search operation using retrieval service.

        Args:
            instance_id: Tool instance ID
            query_list: List of search queries
            retrieval_service_url: URL of the retrieval service
            topk: Number of top results to return
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            concurrent_semaphore=None,
            timeout=timeout,
        )
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return decode_unicode_escapes_only(result_text), metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        timeout = self.timeout
        if isinstance(parameters, str):
            logger.warning(f"parameters is not a dict: {parameters}, type: {type(parameters)}")
        query_from_params = parameters.get("query")
        topk_from_params = parameters.get("topk", self.topk)
        try:
            topk_from_params = int(topk_from_params)
            topk_from_params = min(topk_from_params, 20)
        except Exception:
            topk_from_params = self.topk
        if not query_from_params or not isinstance(query_from_params, str):
            error_msg = "Error: 'query' is missing, empty, or not a string in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return json.dumps({"result": error_msg}, ensure_ascii=False), 0.0, {}

        # Execute search directly
        try:
            result_text, metadata = self.execute_search(
                instance_id, [query_from_params], self.retrieval_service_url, topk_from_params, timeout
            )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Convert metadata to metrics
            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
            }

            return result_text, 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"}, ensure_ascii=False)
            logger.error(f"[SearchTool] Execution failed: {e}")
            return error_result, 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
