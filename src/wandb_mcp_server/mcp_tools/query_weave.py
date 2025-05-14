"""
Raw HTTP-based implementation of the Weave API client.
This module provides the same functionality as the original Weave client-based implementation
but uses raw HTTP requests to interact with the Weave server.
"""

import base64
import calendar
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Import the query models for building complex queries
from weave.trace_server.interface.query import (
    AndOperation,
    ContainsOperation,
    ContainsSpec,
    ConvertOperation,
    EqOperation,
    GetFieldOperator,
    GteOperation,
    GtOperation,
    LiteralOperation,
    NotOperation,
    Query,
)

from wandb_mcp_server.trace_utils import process_traces
from wandb_mcp_server.tools.tools_utils import get_retry_session

os.environ["WANDB_SILENT"] = "True"
logger = logging.getLogger(__name__)

load_dotenv()

QUERY_WEAVE_TRACES_TOOL_DESCRIPTION = """
Query Weave traces, trace metadata, and trace costs with filtering and sorting options.

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

If the users asks for data about "runs" or "experiments" or anything about "experiment tracking"
then use the `query_wandb_gql_tool` instead.
</use_case_selector>

<usage_tips>
query_traces_tool can return a lot of data, below are some usage tips for this function
in order to avoid overwhelming a LLM's context window with too much data.

<managing_llm_context_window>

Returning all weave trace data can possibly result in overwhelming the LLM context window
if there are 100s or 1000s of logged weave traces (depending on how many child traces each has) as
well as resulting in a lot of data from or calls to the weave API.

So, depending on the user query, consider doing the following to return enough data to answer the user query
but not too much data that it overwhelms the LLM context window:

- return only the root traces using the `trace_roots_only` boolean filter if you only need the top-level/parent
traces and don't need the data from all child traces. For example, if a user wants to know the number of
successful traces in a project but doesn't need the data from all child traces. Or if a user
wants to visualise the number of parent traces over time.

- return only the truncated values of the trace data keys in order to first give a preview of the data that can then
inform more targeted weave trace queries from the user. in the extreme you can set `truncate_length` to 0 in order to
only return keys but not the values of the trace data.

- return only the metadata for all the traces (set `metadata_only = True`) if the query doesn't need to know anything
about the structure or content of the individual weave traces. Note that this still requires
requesting all the raw traces data from the weave API so can still result in a lot of data and/or a
lot of calls being made to the weave API.

- return only the columns needed using the `columns` parameter. In weave, the `inputs` and `output` columns of a
trace can contain a lot of data, so avoiding returning these columns can help. Note you have to explicitly specify
the columns you want to return if there are certain columns you don't want to return. Its almost always a good idea to
specficy the columns needed.

<returning_metadata_only>

If `metadata_only = True` this returns only metadata of the traces such as trace counts, token counts,
trace types, time range, status counts and distribution of op names. if `metadata_only = False` the
trace data is returned either in full or truncated to `truncate_length` characters depending if
`return_full_data = True` or `False` respectively.
</returning_metadata_only>

<truncating_trace_data_values>

If `return_full_data = False` the trace data is truncated to `truncate_length` characters,
default 200 characters. Otherwise the trace data is returned in full.
</truncating_trace_data_values>

Remember, LLM context window is precious, only return the minimum amount of data needed to complete an analysis.
</managing_llm_context_window>

<usage_guidance>

- Exploratory queries: For generic exploratory or initial queries about a set of weave traces in a project it can
be a good idea to start with just returning metadata or truncated data. Consider asking the
user for clarification and warn them that returning a lot of weave traces data might
overwhelm the LLM context window. No need to warn them multiple times, just once is enough.

- Project size: Consider using the count_traces_tool to get an estimate of the number of traces in a project
before querying for them as query_trace_tool can return a lot of data.

- Partial op name matching: Use the `op_name_contains` filter if a users has only given a partial op name or if they
are unsure of the exact op name.

- Evaluations: If asked about weave evaluations or evals traces filter for traces with:
    `op_name_contains = "Evaluation.evaluate"` as a first step. These ops are parent traces that contain
    aggregated stats and scores about the evaluation. The child traces of these ops are the actual evaluation results
    for each sample in an evaluation dataset. If asked about individual rows in an evaluation then use the parent_ids
    filter to return the child traces.

- Weave nomenclature: Note that users might refer to weave ops as "traces" or "calls" or "traces" as "ops".

</usage_guidance>
</usage_tips>

Args:
    entity_name: The Weights & Biases entity name (team or username)
    project_name: The Weights & Biases project name
    filters: Dict of filter conditions, supporting:
        - display_name: Filter by display name seen in the Weave UI (string or regex pattern)
        - op_name: Filter by weave op name, a long URI starting with 'weave:///' (string or regex pattern)
        - op_name_contains: Filter for op_name containing this substring (easier than regex)
        - trace_roots_only: Boolean to filter for only top-level/parent traces. Useful when you don't need
            to return the data from all child traces.
        - trace_id: Filter by a specific `trace_id` (e.g., "01958ab9-3c67-7c72-92bf-d023fa5a0d4d").
            A `trace_id` groups multiple calls/spans. Use if the user explicitly say they provided a "trace_id" for a group of operations.
        - call_ids: Filter by specific `call_id`s (also known as Span IDs) (string or list of strings, e.g., ["01958ab9-3c68-7c23-8ccd-c135c7037769"]).
            **GUIDANCE**: `call_id` (Span ID) identifies a *single* operation/span and is typically found in Weave UI URLs.
            If a user provides an ID for a specific item they're viewing, **prefer `call_ids`**.
            Format as a list: `{"call_ids": ["user_provided_id"]}`.
        - parent_ids: Return traces that are children of the given parent trace ids (string or list of strings)
        - status: Filter by trace status, defined as whether or not the trace had an exception or not. Can be
            `success` or `exception`.
        - time_range: Dict with "start" and "end" datetime strings. Datetime strings should be in ISO format
            (e.g. `2024-01-01T00:00:00Z`)
        - attributes: Dict of the weave attributes of the trace.
            Supports nested paths (e.g., "metadata.model_name") via dot notation.
            Value can be:
            *   A literal for exact equality (e.g., `"status": "success"`)
            *   A dictionary with a comparison operator: `$gt`, `$lt`, `$eq`, `$gte`, `$lte` (e.g., `{"token_count": {"$gt": 100}}`)
            *   A dictionary with the `$contains` operator for substring matching on string attributes (e.g., `{"model_name": {"$contains": "gpt-3"}}`)
            **Warning:** The `$contains` operator performs simple substring matching only, full regular expression matching (e.g., via `$regex`) is **not supported** for attributes. Do not attempt to use `$regex`.
        - has_exception: Optional[bool] to filter traces by exception status:
            - None (or key not present): Show all traces regardless of exception status
            - True: Show only traces that have exceptions (exception field is not null)
            - False: Show only traces without exceptions (exception field is null)
    sort_by: Field to sort by (started_at, ended_at, op_name, etc.). Defaults to 'started_at'
    sort_direction: Sort direction ('asc' or 'desc'). Defaults to 'desc'
    limit: Maximum number of results to return. Defaults to None
    include_costs: Include tracked api cost information in the results. Defaults to True
    include_feedback: Include weave annotations (human labels/feedback). Defaults to True
    columns: List of specific columns to include in the results. Its almost always a good idea to specficy the
    columns needed. Defaults to None (all columns).
        Available columns are:
            id: <class 'str'>
            project_id: <class 'str'>
            op_name: <class 'str'>
            display_name: typing.Optional[str]
            trace_id: <class 'str'>
            parent_id: typing.Optional[str]
            started_at: <class 'datetime.datetime'>
            attributes: dict[str, typing.Any]
            inputs: dict[str, typing.Any]
            ended_at: typing.Optional[datetime.datetime]
            exception: typing.Optional[str]
            output: typing.Optional[typing.Any]
            summary: typing.Optional[SummaryMap] # Contains nested data like 'summary.weave.status' and 'summary.weave.latency_ms'
            status: typing.Optional[str] # Synthesized from summary.weave.status if requested
            latency_ms: typing.Optional[int] # Synthesized from summary.weave.latency_ms if requested
            wb_user_id: typing.Optional[str]
            wb_run_id: typing.Optional[str]
            deleted_at: typing.Optional[datetime.datetime]
    expand_columns: List of columns to expand in the results. Defaults to None
    truncate_length: Maximum length for string values in weave traces. Defaults to 200
    return_full_data: Whether to include full untruncated trace data. Defaults to False
    metadata_only: Return only metadata without traces. Defaults to False

Returns:
    JSON string containing either full trace data or metadata only, depending on parameters

<examples>
    ```python
    # Get an overview of the traces in a project
    query_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"root_traces_only": True},
        metadata_only=True,
        return_full_data=False
    )

    # Get failed traces with costs and feedback
    query_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"status": "error"},
        include_costs=True,
        include_feedback=True
    )

    # Get specific columns for traces who's op name (i.e. trace name) contains a specific substring
    query_traces_tool(
        entity_name="my-team",
        project_name="my-project",
        filters={"op_name_contains": "Evaluation.summarize"},
        columns=["id", "op_name", "started_at", "costs"]
    )
    ```
</examples>
"""


def query_traces(
    entity_name: str,
    project_name: str,
    filters: Dict[str, Any] = None,
    sort_by: str = "started_at",
    sort_direction: str = "desc",
    limit: int = None,
    offset: int = 0,
    include_costs: bool = True,
    include_feedback: bool = True,
    columns: List[str] = None,
    expand_columns: List[str] = None,
    api_key: Optional[str] = None,
    query_expr: Optional[Query] = None,
    request_timeout: int = 10,
) -> List[Dict]:
    """Query Weave traces with flexible filtering and sorting options.
    (This version uses direct HTTP requests instead of the Weave client)

    Always ensure that the entity_name and project_name have been provided by the user.

    Parameters
    ----------
    entity_name : str
        The Weights & Biases entity name. This can be a W&B team or username.
    project_name : str
        The Weights & Biases project name.
    filters : Dict[str, Any], optional
        Dict of filter conditions, supporting:
            - display_name: Filter by display name seen in the Weave UI (string or regex pattern)
            - op_name: Filter by weave op name, a long URI starting with 'weave:///' (string or regex pattern)
            - op_name_contains: Filter for op_name containing this substring (easier than regex)
            - trace_id: Filter by a specific `trace_id` (e.g., "01958ab9-3c67-7c72-92bf-d023fa5a0d4d").
                A `trace_id` groups multiple calls/spans. Use if the user explicitly provides a "trace_id" for a group of operations.
            - call_ids: Filter by specific `call_id`s (also known as Span IDs) (string or list of strings, e.g., ["01958ab9-3c68-7c23-8ccd-c135c7037769"]).
                **GUIDANCE**: `call_id` (Span ID) identifies a *single* operation/span and is typically found in Weave UI URLs.
                If a user provides an ID for a specific item they're viewing, **prefer `call_ids`**.
                Format as a list: `{"call_ids": ["user_provided_id"]}`.
            - status: Filter by trace status, can be `success`, `error`, or `running`
            - latency: Filter by latency in milliseconds (summary.weave.latency_ms).
                Use a nested dict with operators: $gt, $lt, $eq, $gte, $lte.
                Note: $lt and $lte are implemented via logical negation.
                e.g., {"latency": {"$gt": 5000}}
            - time_range: Dict with "start" and "end" datetime strings
            - attributes: Dict of attribute path and value/operator to match.
                Supports nested paths (e.g., "metadata.model_name") via dot notation.
                Value can be literal for equality or a dict with operator ($gt, $lt, $eq, $gte, $lte) for comparison
                (e.g., {"token_count": {"$gt": 100}}).
            - has_exception: Optional[bool] to filter traces by exception status:
                - None (or key not present): Show all traces regardless of exception status
                - True: Show only traces that have exceptions (exception field is not null)
                - False: Show only traces without exceptions (exception field is null)
            - trace_roots_only: Boolean to filter for only top-level/parent traces
    sort_by : str, optional
        Field to sort by (e.g., "started_at", "ended_at", "op_name", "summary.weave.latency_ms"). Default is "started_at".
    sort_direction : str, optional
        Sort direction ("asc" or "desc"). Default is "desc".
    limit : int, optional
        Maximum number of results to return.
    offset : int, optional
        Number of results to skip (for pagination). Default is 0.
    include_costs : bool, optional
        Include tracked API cost information in the results. Default is True.
    include_feedback : bool, optional
        Include Weave annotations (e.g., human labels or feedback) information in the results. Default is True.
    columns : List[str], optional
        List of specific columns to include in the results.
    expand_columns : List[str], optional
        List of columns to expand in the results.
    api_key : Optional[str], optional
        The W&B API key. If not provided, will try to get from environment.
    query_expr : Optional[Query], optional
        A pre-built Query object to use instead of building one from filters.
    request_timeout : int, optional
        Timeout for HTTP requests. Default is 10 seconds.

    Returns
    -------
    List[Dict]
        List of traces matching the query parameters.
    """

    if api_key is None:
        api_key = os.environ.get("WANDB_API_KEY")
    project_id = f"{entity_name}/{project_name}"

    # --- Filter Separation ---
    direct_filters_for_callsfilter_model = {}
    complex_filters_for_expr = {}

    if filters:
        # Simple filters for CallsFilter object
        simple_filter_keys = [
            "trace_roots_only",
            "op_names",
            "op_names_prefix",
            "trace_ids",
            "trace_parent_ids",
            "parent_ids",  # kept for backward-compat
            "call_ids",
        ]
        for key in simple_filter_keys:
            if key in filters:
                # Ensure op_names, trace_ids, etc. are lists as expected by CallsFilter
                if key in [
                    "op_names",
                    "op_names_prefix",
                    "trace_ids",
                    "trace_parent_ids",
                    "parent_ids",
                    "call_ids",
                ] and not isinstance(filters[key], list):
                    direct_filters_for_callsfilter_model[key] = [str(filters[key])]
                else:
                    direct_filters_for_callsfilter_model[key] = filters[key]

        # Handle individual op_name and trace_id if op_names/trace_ids not already set
        if (
            "op_name" in filters
            and "op_names" not in direct_filters_for_callsfilter_model
        ):
            # Only add if it's a simple name, not a pattern (patterns go to complex)
            if (
                isinstance(filters["op_name"], str)
                and "*" not in filters["op_name"]
                and ".*" not in filters["op_name"]
            ):
                direct_filters_for_callsfilter_model["op_names"] = [filters["op_name"]]
            else:
                # It's a pattern or complex, send to complex_filters_for_expr
                complex_filters_for_expr["op_name"] = filters["op_name"]
        elif (
            "op_name" in filters and "op_names" in direct_filters_for_callsfilter_model
        ):
            # If op_names is already set, and op_name is a pattern, it needs to go to complex
            if isinstance(filters["op_name"], str) and (
                "*" in filters["op_name"] or ".*" in filters["op_name"]
            ):
                complex_filters_for_expr["op_name"] = filters["op_name"]

        if (
            "trace_id" in filters
            and "trace_ids" not in direct_filters_for_callsfilter_model
        ):
            direct_filters_for_callsfilter_model["trace_ids"] = [
                str(filters["trace_id"])
            ]

        # Complex filters for $expr via _build_query_expression
        # All other keys from the original `filters` dict go here.
        all_handled_direct_keys = set(direct_filters_for_callsfilter_model.keys())
        # Add op_name/trace_id to handled if they were processed into op_names/trace_ids
        if (
            "op_names" in direct_filters_for_callsfilter_model
            and "op_name" in filters
            and filters["op_name"] in direct_filters_for_callsfilter_model["op_names"]
        ):
            all_handled_direct_keys.add("op_name")
        if (
            "trace_ids" in direct_filters_for_callsfilter_model
            and "trace_id" in filters
            and filters["trace_id"] in direct_filters_for_callsfilter_model["trace_ids"]
        ):
            all_handled_direct_keys.add("trace_id")

        for key, value in filters.items():
            if key not in all_handled_direct_keys:
                # Exception: if op_name was simple and handled, but is also in filters, it shouldn't be added again
                if (
                    key == "op_name"
                    and "op_names" in direct_filters_for_callsfilter_model
                    and value in direct_filters_for_callsfilter_model["op_names"]
                    and not ("*" in value or ".*" in value)
                ):
                    continue
                complex_filters_for_expr[key] = value

    # Build the request body
    request_body = {
        "project_id": project_id,
    }

    requested_status_column = False
    requested_latency_ms_column = False
    original_requested_columns = None
    columns_to_send_to_server = columns # Default to original columns

    if columns is not None: # Only modify if columns were specified
        original_requested_columns = list(columns) # Keep a copy of the original request
        columns_to_send_to_server = list(columns) # Start with a mutable copy
        
        # Handle status column
        if "status" in columns_to_send_to_server:
            requested_status_column = True
            columns_to_send_to_server = [col for col in columns_to_send_to_server if col != "status"]
            if "summary" not in columns_to_send_to_server:
                columns_to_send_to_server.append("summary")

        # Handle latency_ms column
        if "latency_ms" in columns_to_send_to_server:
            requested_latency_ms_column = True
            columns_to_send_to_server = [col for col in columns_to_send_to_server if col != "latency_ms"]
            if "summary" not in columns_to_send_to_server:
                columns_to_send_to_server.append("summary")

        # If columns_to_send_to_server became empty because only synthesized columns were requested,
        # ensure 'summary' is fetched.
        if original_requested_columns and not columns_to_send_to_server and (requested_status_column or requested_latency_ms_column):
            columns_to_send_to_server = ["summary"]
        elif not original_requested_columns and (requested_status_column or requested_latency_ms_column):
             # This case should not happen if columns is None initially, but as a safeguard
            columns_to_send_to_server = ["summary"]

    # Add filter if present
    if direct_filters_for_callsfilter_model:
        request_body["filter"] = direct_filters_for_callsfilter_model

    # Add query expression if present
    if query_expr:
        # Use the provided Query object
        request_body["query"] = query_expr.model_dump(by_alias=True)
    else:
        # Build a Query object from the complex filters
        query_expression = _build_query_expression(complex_filters_for_expr)
        if query_expression:
            # Convert the Query object to a dict for the request
            request_body["query"] = query_expression.model_dump(by_alias=True)

    # Add sort criteria if present
    if sort_by:
        request_body["sort_by"] = [{"field": sort_by, "direction": sort_direction}]

    # Add pagination parameters if present
    if limit is not None:
        request_body["limit"] = limit
    if offset is not None:
        request_body["offset"] = offset

    # Add include flags
    request_body["include_costs"] = include_costs
    request_body["include_feedback"] = include_feedback

    # Add columns and expand_columns if present
    # Use columns_to_send_to_server for the actual HTTP request
    if columns_to_send_to_server:
        request_body["columns"] = columns_to_send_to_server
    # If original columns was None (meaning all columns), and we didn't modify it for status,
    # then columns_to_send_to_server is also None, and request_body["columns"] won't be set,
    # which is correct for requesting all columns.
    # If original columns was not None, but columns_to_send_to_server ended up empty 
    # (e.g. original was just ["status"]), it means we are fetching ["summary"] instead.

    if expand_columns:
        request_body["expand_columns"] = expand_columns

    # ----------------- HTTP request -----------------
    weave_server_url = os.environ.get("WEAVE_TRACE_SERVER_URL", "https://trace.wandb.ai")
    url = f"{weave_server_url}/calls/stream_query"

    auth_token = base64.b64encode(f":{api_key}".encode()).decode()

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/jsonl",
        "Authorization": f"Basic {auth_token}",
    }

    session = get_retry_session()

    try:
        response = session.post(
            url,
            headers=headers,
            data=json.dumps(request_body),
            timeout=request_timeout,
            stream=True,
        )

        # Check for errors
        if response.status_code != 200:
            error_msg = f"Error {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # Process the streaming response
        results = []
        for line in response.iter_lines():
            if line:
                # Parse the JSON line
                call_data = json.loads(line.decode("utf-8"))

                # Convert to the same format as the original implementation
                call_dict = {
                    "id": call_data.get("id"),
                    "project_id": call_data.get("project_id"),
                    "op_name": call_data.get("op_name"),
                    "display_name": call_data.get("display_name"),
                    "trace_id": call_data.get("trace_id"),
                    "parent_id": call_data.get("parent_id"),
                    "started_at": call_data.get("started_at"),
                    "ended_at": call_data.get("ended_at"),
                    "inputs": call_data.get("inputs"),
                    "output": call_data.get("output"),
                    "exception": call_data.get("exception"),
                    "attributes": call_data.get("attributes"),
                    "summary": call_data.get("summary"),
                    "wb_run_id": call_data.get("wb_run_id"),
                    "wb_user_id": call_data.get("wb_user_id"),
                    "deleted_at": call_data.get("deleted_at"),
                }

                # Synthesize status if requested and summary is available
                if requested_status_column and "summary" in call_dict:
                    weave_summary = call_dict.get("summary", {}).get("weave", {})
                    call_dict["status"] = weave_summary.get("status") if weave_summary else None
                
                # Synthesize latency_ms if requested and summary is available
                if requested_latency_ms_column and "summary" in call_dict:
                    weave_summary = call_dict.get("summary", {}).get("weave", {})
                    call_dict["latency_ms"] = weave_summary.get("latency_ms") if weave_summary else None

                # Add costs if included
                if include_costs and "costs" in call_data:
                    call_dict["costs"] = call_data.get("costs")

                # Add feedback if included
                if include_feedback and "feedback" in call_data:
                    call_dict["feedback"] = call_data.get("feedback")

                # Column filtering should be based on the *original* user request (original_requested_columns)
                # not columns_to_send_to_server.
                if original_requested_columns:
                    final_call_dict = {}
                    for col_name in original_requested_columns:
                        if col_name in call_dict:
                            final_call_dict[col_name] = call_dict[col_name]
                        # If col_name was 'status' and we synthesized it, it will be included.
                        # If col_name was something else not present in call_dict (e.g. a non-existent field),
                        # it will be omitted, which is standard behavior for column selection.
                    results.append(final_call_dict)
                elif columns is None: # If original columns parameter was None, implies all columns
                    results.append(call_dict)
                # If original_requested_columns was an empty list, results will remain empty unless something is added.
                # This case should be fine as an empty columns list implies user wants nothing.

        return results

    except requests.exceptions.RequestException as e:
        logger.error(
            f"Error executing HTTP request to Weave server for {project_id}: {e}. Request body snippet: {str(request_body)[:1000]}"
        )
        if isinstance(e, requests.exceptions.RetryError):
            if e.__cause__ and hasattr(e.__cause__, 'reason') and e.__cause__.reason:
                logger.error(f"Specific reason for retry exhaustion for {project_id}: {e.__cause__.reason}")
        # traceback.print_exc() # For more detailed debugging if needed
        raise Exception(f"Failed to query Weave traces for {project_id} due to network error: {e}")
    except json.JSONDecodeError as e:
        # This might occur if the streaming response contains an invalid JSON line
        logger.error(
            f"Error decoding JSON line from Weave server stream for {project_id}: {e}. Offending line might be logged by iter_lines if possible."
        )
        raise Exception(f"Failed to parse Weave API streaming response for {project_id}: {e}")
    except Exception as e: # Catch any other unexpected errors
        logger.error(
            f"Unexpected error during HTTP request to Weave server for {project_id}: {e}. Request body snippet: {str(request_body)[:1000]}"
        )
        # traceback.print_exc()
        raise


async def query_paginated_weave_traces(
    entity_name: str,
    project_name: str,
    chunk_size: int = 20,
    filters: Optional[Dict[str, Any]] = None,
    sort_by: str = "started_at",
    sort_direction: str = "desc",
    target_limit: int = None,
    include_costs: bool = True,
    include_feedback: bool = True,
    columns: Optional[List[str]] = None,
    expand_columns: Optional[List[str]] = None,
    truncate_length: Optional[int] = 200,
    return_full_data: bool = False,
    metadata_only: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Query traces with pagination.

    Parameters
    ----------
    entity_name : str
        The Weights & Biases entity name.
    project_name : str
        The Weights & Biases project name.
    chunk_size : int, optional
        The number of traces to fetch per query. Default is 20.
    filters : Optional[Dict[str, Any]], optional
        Filter conditions passed to `query_traces`.
    sort_by : str, optional
        Field to sort by, passed to `query_traces`. Default is "started_at".
    sort_direction : str, optional
        Sort direction, passed to `query_traces`. Default is "desc".
    target_limit : int, optional
        Maximum total number of traces to return.
    include_costs : bool, optional
        Whether to include cost information. Default is True.
    include_feedback : bool, optional
        Whether to include feedback information. Default is True.
    columns : Optional[List[str]], optional
        Specific columns to include in the results.
    expand_columns : Optional[List[str]], optional
        Columns to expand in the results.
    truncate_length : Optional[int], optional
        Length to truncate trace data fields. Default is 200. Set to 0 for metadata_only.
    return_full_data : bool, optional
        Whether to return full, unprocessed trace data. Default is False.
    metadata_only : bool, optional
        If True, only return metadata (counts, etc.) and no trace data. Default is False.
    api_key : Optional[str], optional
        The W&B API key. If not provided, will try to get from environment.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing 'metadata' and optionally 'traces'.
    """
    if api_key is None:
        api_key = os.environ.get("WANDB_API_KEY")

    all_raw_traces = []
    current_offset = 0

    # Parameters that go to query_traces
    query_params = {
        "filters": filters,
        "sort_by": sort_by,
        "sort_direction": sort_direction,
        "include_costs": include_costs,
        "include_feedback": include_feedback,
        "columns": columns,
        "expand_columns": expand_columns,
        "api_key": api_key,
    }
    # Remove None values
    query_params = {k: v for k, v in query_params.items() if v is not None}

    # First, collect all raw traces
    while True:
        logger.info(f"Querying chunk with offset {current_offset}, size {chunk_size}")

        # Calculate chunk size
        remaining = target_limit - len(all_raw_traces) if target_limit else chunk_size
        current_chunk_size = min(chunk_size, remaining) if target_limit else chunk_size

        # Make the query for the current chunk
        chunk_result = query_traces(
            entity_name=entity_name,
            project_name=project_name,
            limit=current_chunk_size,
            offset=current_offset,
            **query_params,
        )

        # Add raw traces to collection
        if not chunk_result:
            break

        all_raw_traces.extend(chunk_result)

        # Check if we should stop
        if len(chunk_result) < current_chunk_size or (
            target_limit and len(all_raw_traces) >= target_limit
        ):
            break

        current_offset += chunk_size

    # Process all traces once, with appropriate parameters based on needs
    processed_result = process_traces(
        traces=all_raw_traces,
        truncate_length=truncate_length
        if not metadata_only
        else 0,  # Only truncate if we need traces
        return_full_data=return_full_data
        if not metadata_only
        else True,  # Use full data for metadata
    )

    result = {"metadata": processed_result["metadata"]}

    # Add traces if needed
    if not metadata_only:
        result["traces"] = (
            processed_result["traces"][:target_limit]
            if target_limit
            else processed_result["traces"]
        )

    return result


def _datetime_to_timestamp(dt_str: str) -> int:
    """Convert an ISO format datetime string to Unix timestamp.

    Parameters
    ----------
    dt_str : str
        The ISO format datetime string. Handles 'Z' suffix for UTC.

    Returns
    -------
    int
        The Unix timestamp (seconds since epoch). Returns 0 if the input is empty or parsing fails.
    """
    if not dt_str:
        return 0

    # Handle 'Z' suffix for UTC
    dt_str = dt_str.replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(dt_str)
        return int(calendar.timegm(dt.utctimetuple()))
    except ValueError:
        # If parsing fails, return 0 (beginning of epoch)
        return 0


def _build_query_expression(filters: Dict[str, Any]) -> Optional[Query]:
    """Build a Query expression from the filter dictionary.

    Parameters
    ----------
    filters : Dict[str, Any]
        Dictionary of filter conditions. See `query_traces` for supported filters.

    Returns
    -------
    Optional[Query]
        The constructed `weave.trace_server.interface.query.Query` object,
        or None if no valid filters requiring a query expression are provided.
    """
    operations = []

    # Define helper to create comparison operations, handling $lt/$lte via $not
    def _create_comparison_op(field_name: str, comparison_dict: dict):
        if not isinstance(comparison_dict, dict) or len(comparison_dict) != 1:
            logger.warning(
                f"Invalid comparison format for {field_name}: {comparison_dict}. Expected dict with one operator key. Skipping."
            )
            return None

        operator, value = next(iter(comparison_dict.items()))

        try:
            field_op_base = GetFieldOperator(**{"$getField": field_name})
            field_op = field_op_base # Default to no $convert

            # Apply $convert selectively as it was found to be needed for some attributes
            if field_name.startswith("attributes."):
                # For general attributes, convert to double if comparing with a number
                if isinstance(value, (int, float)):
                    field_op = ConvertOperation(
                        **{"$convert": {"input": field_op_base, "to": "double"}}
                    )

            literal_op = LiteralOperation(**{"$literal": value})
        except Exception as e:
            logger.warning(
                f"Invalid value for {field_name} comparison {operator}: {value}. Error: {e}. Skipping."
            )
            return None

        if operator == "$gt":
            return GtOperation(**{"$gt": (field_op, literal_op)})
        elif operator == "$gte":
            return GteOperation(**{"$gte": (field_op, literal_op)})
        elif operator == "$eq":
            return EqOperation(**{"$eq": (field_op, literal_op)})
        elif operator == "$lt":  # Implement $lt as $not($gte)
            gte_op = GteOperation(**{"$gte": (field_op, literal_op)})
            return NotOperation(**{"$not": [gte_op]})
        elif operator == "$lte":  # Implement $lte as $not($gt)
            gt_op = GtOperation(**{"$gt": (field_op, literal_op)})
            return NotOperation(**{"$not": [gt_op]})
        else:
            logger.warning(
                f"Unsupported comparison operator '{operator}' for {field_name}. Skipping."
            )
            return None

    # Handle op_name filter (regex or string)
    if "op_name" in filters:
        op_name = filters["op_name"]
        if isinstance(op_name, str):
            # If it's a string with wildcard pattern, treat as contains
            if "*" in op_name or ".*" in op_name:
                # Extract the part between wildcards
                pattern = op_name.replace("*", "").replace(".*", "")
                operations.append(
                    ContainsOperation(
                        **{
                            "$contains": ContainsSpec(
                                input=GetFieldOperator(**{"$getField": "op_name"}),
                                substr=LiteralOperation(**{"$literal": pattern}),
                                case_insensitive=True,
                            )
                        }
                    )
                )
            else:
                # Exact match
                operations.append(
                    EqOperation(
                        **{
                            "$eq": (
                                GetFieldOperator(**{"$getField": "op_name"}),
                                LiteralOperation(**{"$literal": op_name}),
                            )
                        }
                    )
                )
        elif hasattr(op_name, "pattern"):  # Regex pattern
            operations.append(
                ContainsOperation(
                    **{
                        "$contains": ContainsSpec(
                            input=GetFieldOperator(**{"$getField": "op_name"}),
                            substr=LiteralOperation(**{"$literal": op_name.pattern}),
                            case_insensitive=True,
                        )
                    }
                )
            )

    # Handle op_name_contains custom filter (for simple substring matching)
    if "op_name_contains" in filters:
        substring = filters["op_name_contains"]
        operations.append(
            ContainsOperation(
                **{
                    "$contains": ContainsSpec(
                        input=GetFieldOperator(**{"$getField": "op_name"}),
                        substr=LiteralOperation(**{"$literal": substring}),
                        case_insensitive=True,
                    )
                }
            )
        )

    # Handle display_name filter (regex or string)
    if "display_name" in filters:
        display_name = filters["display_name"]
        if isinstance(display_name, str):
            # If it's a string with wildcard pattern, treat as contains
            if "*" in display_name or ".*" in display_name:
                # Extract the part between wildcards
                pattern = display_name.replace("*", "").replace(".*", "")
                operations.append(
                    ContainsOperation(
                        **{
                            "$contains": ContainsSpec(
                                input=GetFieldOperator(**{"$getField": "display_name"}),
                                substr=LiteralOperation(**{"$literal": pattern}),
                                case_insensitive=True,
                            )
                        }
                    )
                )
            else:
                # Exact match
                operations.append(
                    EqOperation(
                        **{
                            "$eq": (
                                GetFieldOperator(**{"$getField": "display_name"}),
                                LiteralOperation(**{"$literal": display_name}),
                            )
                        }
                    )
                )
        elif hasattr(display_name, "pattern"):  # Regex pattern
            operations.append(
                ContainsOperation(
                    **{
                        "$contains": ContainsSpec(
                            input=GetFieldOperator(**{"$getField": "display_name"}),
                            substr=LiteralOperation(
                                **{"$literal": display_name.pattern}
                            ),
                            case_insensitive=True,
                        )
                    }
                )
            )

    # Handle display_name_contains custom filter (for simple substring matching)
    if "display_name_contains" in filters:
        substring = filters["display_name_contains"]
        operations.append(
            ContainsOperation(
                **{
                    "$contains": ContainsSpec(
                        input=GetFieldOperator(**{"$getField": "display_name"}),
                        substr=LiteralOperation(**{"$literal": substring}),
                        case_insensitive=True,
                    )
                }
            )
        )

    # Handle status filter based on summary.weave.status using dot notation
    if "status" in filters:
        target_status = filters["status"]
        if isinstance(target_status, str):
            comp_op = _create_comparison_op(
                "summary.weave.status", {"$eq": target_status.lower()}
            )
            if comp_op:
                operations.append(comp_op)
        else:
            logger.warning(
                f"Invalid status filter value: {target_status}. Expected a string. Skipping."
            )

    # Handle time range filter (convert ISO datetime strings to Unix seconds)
    if "time_range" in filters:
        time_range = filters["time_range"]

        # >= start
        if "start" in time_range and time_range["start"]:
            start_ts = _datetime_to_timestamp(time_range["start"])
            if start_ts > 0:
                comp = _create_comparison_op("started_at", {"$gte": start_ts})
                if comp:
                    operations.append(comp)

        # < end (i.e. started_at strictly before end_ts)
        if "end" in time_range and time_range["end"]:
            end_ts = _datetime_to_timestamp(time_range["end"])
            if end_ts > 0:
                comp = _create_comparison_op("started_at", {"$lt": end_ts})
                if comp:
                    operations.append(comp)

    # Handle wb_run_id filter (top-level)
    if "wb_run_id" in filters:
        run_id = filters["wb_run_id"]
        # This filter expects a string for wb_run_id and uses $contains or $eq.
        # It should not be converted to double.
        if isinstance(run_id, str):
            if (
                "$contains" in run_id or "*" in run_id
            ):  # Simple check for contains style
                pattern = run_id.replace("$contains:", "").replace(
                    "*", ""
                )  # Basic cleanup
                operations.append(
                    ContainsOperation(
                        **{
                            "$contains": ContainsSpec(
                                input=GetFieldOperator(**{"$getField": "wb_run_id"}),
                                substr=LiteralOperation(
                                    **{"$literal": pattern.strip()}
                                ),
                                case_insensitive=True,
                            )
                        }
                    )
                )
            else:
                operations.append(
                    EqOperation(
                        **{
                            "$eq": (
                                GetFieldOperator(**{"$getField": "wb_run_id"}),
                                LiteralOperation(**{"$literal": run_id}),
                            )
                        }
                    )
                )
        elif (
            isinstance(run_id, dict) and "$contains" in run_id
        ):  # wb_run_id: {"$contains": "foo"}
            pattern = run_id["$contains"]
            if isinstance(pattern, str):
                operations.append(
                    ContainsOperation(
                        **{
                            "$contains": ContainsSpec(
                                input=GetFieldOperator(**{"$getField": "wb_run_id"}),
                                substr=LiteralOperation(**{"$literal": pattern}),
                                case_insensitive=True,
                            )
                        }
                    )
                )
            else:
                logger.warning(
                    f"Invalid $contains value for wb_run_id: {pattern}. Expected string."
                )
        else:
            logger.warning(
                f"Invalid wb_run_id filter value: {run_id}. Expected a string or dict with $contains. Skipping."
            )

    # Handle latency filter based on summary.weave.latency_ms
    if "latency" in filters:
        latency_filter = filters["latency"]
        comp_op = _create_comparison_op("summary.weave.latency_ms", latency_filter)
        if comp_op:
            operations.append(comp_op)

    # Handle attributes filter using dot notation AND supporting comparison operators
    if "attributes" in filters:
        attributes_filters = filters["attributes"]
        if isinstance(attributes_filters, dict):
            for attr_path, attr_value_or_op in attributes_filters.items():
                full_attr_path = f"attributes.{attr_path}"

                # Check if the value is a comparison operator dict or a literal
                if (
                    isinstance(attr_value_or_op, dict)
                    and len(attr_value_or_op) == 1
                    and next(iter(attr_value_or_op.keys()))
                    in ["$gt", "$gte", "$lt", "$lte", "$eq"]
                ):
                    # It's a comparison operation
                    comp_op = _create_comparison_op(full_attr_path, attr_value_or_op)
                    if comp_op:
                        operations.append(comp_op)
                else:
                    # Assume literal equality
                    comp_op = _create_comparison_op(
                        full_attr_path, {"$eq": attr_value_or_op}
                    )
                    if comp_op:
                        operations.append(comp_op)

                # Check for $contains operation in attributes
                if (
                    isinstance(attr_value_or_op, dict)
                    and "$contains" in attr_value_or_op
                ):
                    if isinstance(attr_value_or_op["$contains"], str):
                        contains_op = ContainsOperation(
                            **{
                                "$contains": ContainsSpec(
                                    input=GetFieldOperator(
                                        **{"$getField": full_attr_path}
                                    ),
                                    substr=LiteralOperation(
                                        **{"$literal": attr_value_or_op["$contains"]}
                                    ),
                                    case_insensitive=True,
                                )
                            }
                        )
                        operations.append(contains_op)
                    else:
                        logger.warning(
                            f"Invalid value for $contains on {full_attr_path}: {attr_value_or_op['$contains']}. Expected string."
                        )
        else:
            logger.warning(
                f"Invalid format for 'attributes' filter: {attributes_filters}. Expected a dictionary. Skipping."
            )

    # Handle has_exception filter (checking top-level exception field)
    if "has_exception" in filters:
        has_exception = filters["has_exception"]
        # Skip filtering if has_exception is None (show everything)
        if has_exception is not None:
            # Create base operation that checks if exception is None (no exception case)
            base_op = EqOperation(
                **{
                    "$eq": (
                        GetFieldOperator(**{"$getField": "exception"}),
                        LiteralOperation(**{"$literal": None}),
                    )
                }
            )

            if has_exception:
                # For has_exception=True: Negate the operation to get NOT NULL
                operations.append(NotOperation(**{"$not": [base_op]}))
            else:
                # For has_exception=False: Use the operation as is
                operations.append(base_op)

    # Combine all operations with AND
    if operations:
        if len(operations) == 1:
            # Wrap the single operation in the Query model structure
            return Query(**{"$expr": operations[0]})
        else:
            # Wrap the AndOperation in the Query model structure
            and_op = AndOperation(**{"$and": operations})
            return Query(**{"$expr": and_op})

    return None  # No complex filters, so no Query object needed for the 'query' arg
