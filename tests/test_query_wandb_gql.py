# noqa: D100
"""Integration tests that verify Anthropic selects `query_wandb_gql_tool`.

These tests send natural-language questions about the W&B *Models* data for the
`wandb-applied-ai-team/mcp-tests` project.  The Anthropic model should respond
with a `tool_use` invoking `query_wandb_gql_tool`, which we then execute and
validate.
"""

import json
import logging
import os
from typing import Any, Dict, List
import sys

import pytest
from rich.logging import RichHandler
from rich.console import Console
from wandb_mcp_server.tools.query_wandb_gql import (
    QUERY_WANDB_GQL_TOOL_DESCRIPTION,
    query_paginated_wandb_gql,
)
from wandb_mcp_server.tools.tools_utils import generate_anthropic_tool_schema
from tests.anthropic_test_utils import (
    call_anthropic,
    extract_anthropic_tool_use,
    check_correctness_tool,
)

import weave


# Root logging configuration
_console = Console(file=sys.__stdout__, force_terminal=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[RichHandler(show_time=True, show_level=True, show_path=False, markup=True, console=_console)],
    force=True,
)

logger = logging.getLogger(__name__)

# weave.init("wandb-applied-ai-team/wandb-mcp-server-test-outputs")
# os.environ["WANDB_SILENT"] = "true"

# -----------------------------------------------------------------------------
# Environment guards
# -----------------------------------------------------------------------------

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not WANDB_API_KEY:
    pytest.skip(
        "WANDB_API_KEY environment variable not set; skipping live GraphQL tests.",
        allow_module_level=True,
    )
if not ANTHROPIC_API_KEY:
    pytest.skip(
        "ANTHROPIC_API_KEY environment variable not set; skipping Anthropic tests.",
        allow_module_level=True,
    )

# -----------------------------------------------------------------------------
# Static test context
# -----------------------------------------------------------------------------

TEST_WANDB_ENTITY = "wandb-applied-ai-team"
TEST_WANDB_PROJECT = "mcp-tests"
MODEL_NAME = "claude-3-7-sonnet-20250219"
CORRECTNESS_MODEL_NAME = "claude-3-5-haiku-20241022"

# -----------------------------------------------------------------------------
# Build tool schema for Anthropic
# -----------------------------------------------------------------------------

available_tools: Dict[str, Dict[str, Any]] = {
    "query_paginated_wandb_gql": {
        "function": query_paginated_wandb_gql,
        "schema": generate_anthropic_tool_schema(
            func=query_paginated_wandb_gql,
            description=QUERY_WANDB_GQL_TOOL_DESCRIPTION,
        ),
    }
}

tools: List[Dict[str, Any]] = [available_tools["query_paginated_wandb_gql"]["schema"]]

# -----------------------------------------------------------------------------
# Compute baseline runCount once so that tests have a stable expected value
# -----------------------------------------------------------------------------

BASELINE_QUERY = (
    """
query ProjectRunCount($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    runCount
  }
}
"""
)
BASELINE_VARIABLES = {"entity": TEST_WANDB_ENTITY, "project": TEST_WANDB_PROJECT}

# Compute baseline
logger.info("Fetching baseline runCount for %s/%s", TEST_WANDB_ENTITY, TEST_WANDB_PROJECT)
_baseline_result = query_paginated_wandb_gql(BASELINE_QUERY, BASELINE_VARIABLES)
BASELINE_RUN_COUNT: int = _baseline_result["project"]["runCount"]
logger.info("Baseline runCount = %s", BASELINE_RUN_COUNT)

# -----------------------------------------------------------------------------
# Natural-language queries to test
# -----------------------------------------------------------------------------

test_queries = [
    {
        "question": "How many runs are currently logged in the `{project_name}` project under the `{entity_name}` entity?",
        "expected_output": 37,
    },
    {
        "question": "What's the total experiment count for `{entity_name}/{project_name}`?",
        "expected_output": 37,
    },
    {
        "question": "In `{project_name}` in entity `{entity_name}` how many runs were run on April 29th 2025?",
        "expected_output": 37,
    },
    {
        "question": "Could you report the number of tracked runs in `{entity_name}/{project_name}` with lr 0.002?",
        "expected_output": 7,
    },
    {
        "question": "what was the run with the best eval loss in the `{project_name}` project belonging to `{entity_name}`.",
        "expected_output": "run_id: h0fm5qp5 OR run_name: transformer_7_bs-128_lr-0.008_5593616",
    },
    {
        "question": "How many steps in run gtng2y4l `{entity_name}/{project_name}` right now.",
        "expected_output": 750000,
    },
    {
        "question": "How many steps in run transformer_25_bs-33554432_lr-0.026000000000000002_2377215 `{entity_name}/{project_name}` right now.",
        "expected_output": 750000,
    },
    {
        "question": "What's the batch size of the run with best evaluation accuracy for `{project_name}` inside `{entity_name}`?",
        "expected_output": 16,
    },
    # {
    #     "question": "Count the runs in my `{entity_name}` entity for the `{project_name}` project.",
    #     "expected_output": BASELINE_RUN_COUNT,
    # },
    # {
    #     "question": "How big is the experiment set for `{entity_name}/{project_name}`?",
    #     "expected_output": BASELINE_RUN_COUNT,
    # },
    # {
    #     "question": "Tell me the number of runs tracked in `{project_name}` (entity `{entity_name}`).",
    #     "expected_output": BASELINE_RUN_COUNT,
    # },
]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample",
    test_queries,
    ids=[f"sample_{i}" for i, _ in enumerate(test_queries)],
)
def test_query_wandb_gql(sample):
    """End-to-end test: NL question → Anthropic → tool_use → result validation."""

    query_text = sample["question"].format(
        entity_name=TEST_WANDB_ENTITY,
        project_name=TEST_WANDB_PROJECT,
    )
    expected_output = sample["expected_output"]

    logger.info("\n==============================")
    logger.info("QUERY: %s", query_text)

    # --- Retry Logic Setup ---
    max_retries = 1
    last_reasoning = "No correctness check performed yet."
    last_is_correct = False
    first_call_assistant_response = None # Store the response dict from the first model
    tool_result = None # Store the result of executing the tool

    # Initial messages for the first attempt
    messages_first_call = [{"role": "user", "content": query_text}]

    for attempt in range(max_retries + 1):
        logger.info(f"\n--- Attempt {attempt + 1} / {max_retries + 1} ---")

        if attempt > 0:
            # We are retrying. Add the previous assistant response and a user message with feedback.
            if first_call_assistant_response:
                messages_first_call.append(first_call_assistant_response) # Add previous assistant message (contains tool use)
            else:
                # Should not happen in retry logic, but defensively handle
                 logger.warning("Attempting retry, but no previous assistant response found.")

            # Construct the user message asking for a retry
            retry_user_message_content = f"""
Executing the previous tool call resulted in:
```json
{json.dumps(tool_result, indent=2)}
```
A separate check determined this result was incorrect for the original query.
The reasoning provided was: "{last_reasoning}".

Please re-analyze the original query ("{query_text}") and the result from your previous attempt, then try generating the 'query_paginated_wandb_gql' tool call again.
"""
            messages_first_call.append({"role": "user", "content": retry_user_message_content})

        # --- First Call: Get the query_paginated_wandb_gql tool use ---
        response = call_anthropic(
            model_name=MODEL_NAME,
            messages=messages_first_call,
            tools=tools, # Provide the GQL tool schema
        )
        first_call_assistant_response = response # Store this response for potential next retry
        _, tool_name, tool_input, _ = extract_anthropic_tool_use(response)

        logger.info(f"Attempt {attempt + 1}: Tool emitted by model: {tool_name}")
        logger.info(f"Attempt {attempt + 1}: Tool input: {json.dumps(tool_input, indent=2)}")

        assert tool_name == "query_paginated_wandb_gql", f"Attempt {attempt + 1}: Expected 'query_paginated_wandb_gql', got '{tool_name}'"

        # --- Execute the GQL tool ---
        try:
            tool_result = available_tools[tool_name]["function"](**tool_input)
            logger.info(f"Attempt {attempt + 1}: Tool result: {json.dumps(tool_result, indent=2)}") # Log full result
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: Error executing tool '{tool_name}' with input {tool_input}: {e}", exc_info=True)
            pytest.fail(f"Attempt {attempt + 1}: Tool execution failed: {e}")


        # --- Second Call: Perform Correctness Check (Separate Task) ---
        logger.info(f"\n--- Starting Correctness Check for Attempt {attempt + 1} ---")

        try:
            # Prepare the prompt for the check - provide all context clearly
            correctness_prompt = f"""
            Please evaluate if the provided 'Actual Tool Result' correctly addresses the 'Original User Query' and seems consistent with the 'Expected Output'. Use the 'check_correctness_tool' to provide your reasoning and conclusion.

            Original User Query:
            "{query_text}"

            Expected Output (for context, may not be directly comparable in structure):

            {json.dumps(expected_output, indent=2)}

            Actual Tool Result from 'query_paginated_wandb_gql':

            {json.dumps(tool_result, indent=2)}

            """

            messages_check_call = [{"role": "user", "content": correctness_prompt}]
            correctness_response = call_anthropic(
                model_name=CORRECTNESS_MODEL_NAME,
                messages=messages_check_call,
                check_correctness_tool=check_correctness_tool
            )

            logger.info(f"Attempt {attempt + 1}: Correctness check response:\n{correctness_response}\n\n")

            # --- Extract and Validate Correctness Tool Use ---
            _, check_tool_name, check_tool_input, _ = extract_anthropic_tool_use(correctness_response)

            assert check_tool_name == "check_correctness_tool", f"Attempt {attempt + 1}: Expected correctness tool, got {check_tool_name}"
            assert "reasoning" in check_tool_input, f"Attempt {attempt + 1}: Correctness tool missing 'reasoning'"
            assert "is_correct" in check_tool_input, f"Attempt {attempt + 1}: Correctness tool missing 'is_correct'"

            # 2. Extract the data from the input dictionary
            try:
                reasoning_text = check_tool_input['reasoning']
                is_correct_flag = check_tool_input['is_correct']

                # Store the latest results
                last_reasoning = reasoning_text
                last_is_correct = is_correct_flag

                logger.info(f"Attempt {attempt + 1}: Correctness Reasoning: {reasoning_text}")
                logger.info(f"Attempt {attempt + 1}: Is Correct according to LLM: {is_correct_flag}")

                if is_correct_flag:
                    logger.info(f"--- Correctness check passed on attempt {attempt + 1}. ---")
                    break # Exit the loop successfully

                # If not correct, and this is the last attempt, the loop will end naturally.

            except KeyError as e:
                logger.error(f"Attempt {attempt + 1}: Missing expected key in correctness tool input: {e}")
                logger.error(f"Attempt {attempt + 1}: Full input received: {check_tool_input}")
                # Store failure info before failing
                last_is_correct = False
                last_reasoning = f"Correctness tool response missing key: {e}"
                pytest.fail(f"Attempt {attempt + 1}: Correctness tool response was missing key: {e}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Error processing correctness tool input: {e}", exc_info=True)
                # Store failure info before failing
                last_is_correct = False
                last_reasoning = f"Failed to process correctness tool input: {e}"
                pytest.fail(f"Attempt {attempt + 1}: Failed to process correctness tool input: {e}")

        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: Error during correctness check for query '{query_text}': {e}", exc_info=True)
            # Store failure info before failing
            last_is_correct = False
            last_reasoning = f"Correctness check failed with exception: {e}"
            pytest.fail(f"Attempt {attempt + 1}: Correctness check failed with exception: {e}")

    # --- After the loop, fail the test if the last attempt wasn't correct ---
    if not last_is_correct:
        pytest.fail(f"LLM evaluation failed after {max_retries + 1} attempts. "
                    f"Final is_correct_flag is `{last_is_correct}`. "
                    f"Final Reasoning: '{last_reasoning}'")

    # If we reach here, it means the correctness check passed within the allowed attempts.
    logger.info("--- Test passed within allowed attempts. ---")

    # --- Removed the old assertion logic that was outside the try/loop block --- 