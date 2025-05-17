import logging
import traceback
from typing import Any, Dict

from weave.trace_server import trace_server_interface
from weave.trace_server.interface.query import Query
from wandb_mcp_server.query_weave import get_weave_trace_server, get_args, _build_query_expression

logger = logging.getLogger(__name__)

COUNT_WEAVE_TRACES_TOOL_DESCRIPTION = """count Weave traces matching the given filters.

Use this tool to query data from Weights & Biases Weave, an observability product for 
tracing and evaluating LLMs and GenAI apps.

This tool only provides COUNT information about traces, not actual metrics or run data.

<tool_choice_guidance>
<wandb_vs_weave_product_distinction>
**IMPORTANT PRODUCT DISTINCTION:**
W&B offers two distinct products with different purposes:

1. W&B Models: A system for ML experiment tracking, hyperparameter optimization, and model 
    lifecycle management. Use `query_wandb_gql_tool` for questions about:
    - Experiment runs, metrics, and performance comparisons
    - Artifact management and model registry
    - Hyperparameter optimization and sweeps
    - Project dashboards and reports

2. W&B Weave: A toolkit for LLM and GenAI application observability and evaluation. Use
    `query_weave_traces_tool` (this tool) for questions about:
    - Execution traces and paths of LLM operations
    - LLM inputs, outputs, and intermediate results
    - Chain of thought visualization and debugging
    - LLM evaluation results and feedback
</wandb_vs_weave_product_distinction>

<use_case_selector>
**USE CASE SELECTOR - READ FIRST:**
- For runs, metrics, experiments, artifacts, sweeps etc → use query_wandb_gql_tool
- For traces, LLM calls, chain-of-thought, LLM evaluations, AI agent traces, AI apps etc → use query_weave_traces_tool

=====================================================================
⚠️ TOOL SELECTION WARNING ⚠️
This tool is ONLY for WEAVE TRACES (LLM operations), NOT for run metrics or experiments!
=====================================================================

**KEYWORD GUIDE:**
If user question contains:
- "runs", "experiments", "metrics" → Use query_wandb_gql_tool
- "traces", "LLM calls" etc → Use this tool

**COMMON MISUSE CASES:**
❌ "Looking at metrics of my latest runs" - Do NOT use this tool, use query_wandb_gql_tool instead
❌ "Compare performance across experiments" - Do NOT use this tool, use query_wandb_gql_tool instead
</use_case_selector>
</tool_choice_guidance>

Returns the total number of traces in a project and the number of root
(i.e. "parent" or top-level) traces.

This is more efficient than query_trace_tool when you only need the count.
This can be useful to understand how many traces are in a project before
querying for them as query_trace_tool can return a lot of data.

Parameters
----------
entity_name : str
    The Weights & Biases entity name (team or username).
project_name : str
    The Weights & Biases project name.
filters : Dict[str, Any], optional
    Dict of filter conditions, supporting:
        - display_name: Filter by display name (string or regex pattern)
        - op_name_contains: Filter for op_name containing a substring. Not a good idea to use in conjunction with trace_roots_only.
        - trace_id: Filter by specific trace ID
        - status: Filter by trace status ('success', 'error', etc.)
        - time_range: Dict with "start" and "end" datetime strings
        - latency: Filter by latency in milliseconds (summary.weave.latency_ms).
            Use a nested dict with operators: $gt, $lt, $eq, $gte, $lte.
            ($lt and $lte are implemented via logical negation on the backend).
            e.g., {"latency": {"$gt": 5000}}
        - attributes: Dict of attribute path and value/operator to match.
            Supports nested paths (e.g., "metadata.model_name") via dot notation.
            Value can be literal for equality or a dict with operator ($gt, $lt, $eq, $gte, $lte) for comparison
            (e.g., {"token_count": {"$gt": 100}}).
        - has_exception: Boolean to filter traces with/without exceptions
        - trace_roots_only: Boolean to filter for only top-level (aka parent) traces

Returns
-------
int
    The number of traces matching the query parameters.

Examples
--------
>>> # Count failed traces
>>> count = count_traces(
...     entity_name="my-team",
...     project_name="my-project",
...     filters={"status": "error"}
... )
>>> # Count traces faster than 500ms
>>> count = count_traces(
...     entity_name="my-team",
...     project_name="my-project",
...     filters={"latency": {"$lt": 500}}
... )
"""

def count_traces(
    entity_name: str, project_name: str, filters: Dict[str, Any] = None
) -> int:
    """Count the number of traces matching the given filters.

    Counts without retrieving the full trace data, making it more efficient
    than `query_traces` when only the count is needed.

    Parameters
    ----------
    entity_name : str
        The Weights & Biases entity name (team or username).
    project_name : str
        The Weights & Biases project name.
    filters : Dict[str, Any], optional
        Dict of filter conditions, supporting:
            - display_name: Filter by display name (string or regex pattern)
            - op_name_contains: Filter for op_name containing a substring
            - trace_id: Filter by specific trace ID
            - status: Filter by trace status ('success', 'error', etc.)
            - latency: Filter by latency in milliseconds (summary.weave.latency_ms).
                Use a nested dict with operators: $gt, $lt, $eq, $gte, $lte.
                Note: $lt and $lte are implemented via logical negation.
                e.g., {"latency": {"$gt": 5000}}
            - time_range: Dict with "start" and "end" datetime strings
            - attributes: Dict of attribute path and value/operator to match.
                Supports nested paths (e.g., "metadata.model_name") via dot notation.
                Value can be literal for equality or a dict with operator ($gt, $lt, $eq, $gte, $lte) for comparison
                (e.g., {"token_count": {"$gt": 100}}).
            - has_exception: Boolean to filter traces with/without exceptions
            - trace_roots_only: Boolean to filter for only top-level (aka parent) traces

    Returns
    -------
    int
        The number of traces matching the query parameters.

    Examples
    --------
    >>> # Count failed traces
    >>> count = count_traces(
    ...     entity_name="my-team",
    ...     project_name="my-project",
    ...     filters={"status": "error"}
    ... )
    >>> # Count traces matching an attribute and latency > 1s
    >>> count = count_traces(
    ...     entity_name="my-team",
    ...     project_name="my-project",
    ...     filters={
    ...         "attributes": {"metadata.environment": "production"},
    ...         "latency": {"$gt": 1000}
    ...     }
    ... )
    """
    args = get_args()
    project_id = f"{entity_name}/{project_name}"

    trace_server = get_weave_trace_server(args.wandb_api_key, project_id)

    # Build the filter from the provided filter dictionary
    calls_filter = None
    query_expr = None

    if filters:
        # Initialize filter with trace_roots_only if specified
        trace_roots_only = filters.get("trace_roots_only", False)
        calls_filter = trace_server_interface.CallsFilter(
            trace_roots_only=trace_roots_only
        )

        # Direct filter handling for CallsFilter
        if "op_names" in filters or "op_name" in filters:
            op_names = []
            if "op_names" in filters:
                op_names_filter = filters["op_names"]
                if isinstance(op_names_filter, list):
                    op_names.extend(op_names_filter)
                else:
                    op_names.append(op_names_filter)

            if "op_name" in filters and isinstance(filters["op_name"], str):
                op_names.append(filters["op_name"])

            if op_names:
                calls_filter.op_names = op_names

        if "trace_ids" in filters:
            trace_ids = (
                filters["trace_ids"]
                if isinstance(filters["trace_ids"], list)
                else [filters["trace_ids"]]
            )
            calls_filter.trace_ids = trace_ids

        if "trace_id" in filters:
            trace_id = filters["trace_id"]
            calls_filter.trace_ids = [trace_id]

        if "parent_ids" in filters:
            parent_ids = (
                filters["parent_ids"]
                if isinstance(filters["parent_ids"], list)
                else [filters["parent_ids"]]
            )
            calls_filter.parent_ids = parent_ids

        if "call_ids" in filters:
            call_ids = (
                filters["call_ids"]
                if isinstance(filters["call_ids"], list)
                else [filters["call_ids"]]
            )
            calls_filter.call_ids = call_ids

        # Build the more complex query expression
        query_expr = _build_query_expression(filters)

    # Create a CallsQueryStatsReq object
    logger.debug(f"Calls filter: {calls_filter}")
    logger.debug(f"Query expression: {query_expr}")
    stats_req = trace_server_interface.CallsQueryStatsReq(
        project_id=project_id,
        filter=calls_filter,
        query=Query(**{"$expr": query_expr}) if query_expr else None,
    )

    # Execute the query and get the count
    stats_res = trace_server.calls_query_stats(stats_req)
    
    return stats_res.count
