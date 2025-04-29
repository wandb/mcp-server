import logging
import os

import anthropic
import pytest
from rich.logging import RichHandler

from wandb_mcp_server.tools.count_traces import (
    COUNT_WEAVE_TRACES_TOOL_DESCRIPTION,
    count_traces,
)
from wandb_mcp_server.tools.tools_utils import generate_anthropic_tool_schema

logger = logging.getLogger(__name__)

rich_handler = RichHandler(
    level=logging.DEBUG,
    show_time=True,
    show_level=True,
    show_path=False,
    markup=True, 
)

# Remove existing handlers if any and add RichHandler
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(rich_handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Prevent duplicate logs if root logger is configured

os.environ["WANDB_SILENT"] = "true"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    pytest.skip(
        "ANTHROPIC_API_KEY environment variable not set; skipping live Anthropic tests.",
        allow_module_level=True,
    )

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

TEST_WANDB_ENTITY = "wandb-applied-ai-team"  # "c-metrics"
TEST_WANDB_PROJECT = "mcp-tests"  # "hallucination"
model_name = "claude-3-7-sonnet-20250219"

available_tools = {
    "count_traces": {
        "function": count_traces,
        "schema": generate_anthropic_tool_schema(
            func=count_traces, description=COUNT_WEAVE_TRACES_TOOL_DESCRIPTION
        ),
    }
}

tools = [available_tools["count_traces"]["schema"]]


test_queries = [
    {
        "question": "Please count the total number of traces recorded in the `{project_name}` project under the `{entity_name}` entity.",
        "expected_output": 21639,
    },
    {
        "question": "How many Weave call logs exist for the `{project_name}` project in my `{entity_name}` entity?",  # (Uses "call logs" instead of "traces")
        "expected_output": 21639,
    },
    {
        "question": "What's the volume of traces for `{project_name}` in the `{entity_name}` entity?",  # (Assumes default entity or requires clarification, implies counting)
        "expected_output": 21639,
    },
    {
        "question": "Count the calls that resulted in an error within the `{entity_name}/{project_name}` project.",  # (Requires filtering by status='error')
        "expected_output": 136,
    },
    {
        "question": "How many times has the `generate_joke` operation been invoked in the `{project_name}` project for the `{entity_name}`?",  # (Requires filtering by op_name)
        "expected_output": 4,
    },
    {
        "question": "The date is March 12th, 2025. Give me the parent trace count for `{entity_name}/{project_name}` last month.",  # (Requires calculating and applying a time filter)
        "expected_output": 262,
    },
    {
        "question": "Can you count the parent traces in `{entity_name}/{project_name}`?",  # (Requires, root traces)
        "expected_output": 475,
    },
    {
        "question": "`{entity_name}/{project_name}` trace tally?",  # (Requires inferring the need for counting and likely asking for the entity)
        "expected_output": 21639,
    },
    {
        "question": "How many traces in `{entity_name}/{project_name}` took more than 10 minutes to run?",  # (Requires an attribute filter)
        "expected_output": 155,
    },
    {
        "question": "How many traces in `{entity_name}/{project_name}` took less than 2 seconds to run?",  # (Requires an attribute filter)
        "expected_output": 12357,
    },
    {
        "question": "THe date is April 20th, 2025. Count failed traces for the `openai.chat.completions` op within the `{entity_name}/{project_name}` project since the 27th of February 2025 up to March 1st..",  #  (Requires combining status='success', trace_roots_only=True, op_name, and a time filter)
        "expected_output": 15,
    },
]

# Force claude to use a tool: tool_choice = {"type": "tool", "name": "get_weather"}


def call_anthropic(model_name, messages, tools):
    response = client.messages.create(
        model=model_name, max_tokens=4000, tools=tools, messages=messages
    )
    return response


def extract_anthropic_tool_use(response) -> tuple:
    tool_use = None
    for idx, content in enumerate(response.content):
        logger.debug(f"Response content {idx}: {content}")
        if content.type == "tool_use":
            tool_use = content
            break

    if tool_use:
        tool_name = tool_use.name
        tool_input = tool_use.input
        tool_id = tool_use.id

        return tool_use, tool_name, tool_input, tool_id
    else:
        return None, None, None, None


def get_anthropic_tool_result_message(tool_result, tool_id) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": str(tool_result)}
        ],
    }


# -----------------------
# Pytest integration
# -----------------------


@pytest.mark.parametrize(
    "sample", test_queries, ids=[f"sample_{i}" for i, _ in enumerate(test_queries)]
)
def test_count_traces(sample):
    """Run each natural-language query end-to-end through the Anthropic model and
    verify that the invoked tool returns the expected value."""

    query_text = sample["question"].format(
        entity_name=TEST_WANDB_ENTITY,
        project_name=TEST_WANDB_PROJECT,
    )
    expected_output = sample["expected_output"]

    logger.info("==============================")
    logger.info(f"QUERY: {query_text}")

    messages = [{"role": "user", "content": query_text}]

    response = call_anthropic(model_name, messages, tools)
    _, tool_name, tool_input, _ = extract_anthropic_tool_use(response)

    logger.info(f"Tool emitted by model: {tool_name}")
    logger.debug(f"Tool input: {tool_input}")

    assert tool_name is not None, "Model did not emit a tool call"

    # Execute the real tool — no mocking.
    tool_result = available_tools[tool_name]["function"](**tool_input)

    logger.info(f"Tool result: {tool_result} (expected {expected_output})")

    assert tool_result == expected_output, (
        f"Unexpected result for query `{query_text}`: {tool_result} (expected {expected_output})"
    )
