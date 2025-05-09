import os
import calendar
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import weave
from dotenv import load_dotenv
from weave.trace_server import trace_server_interface
from weave.trace_server.interface.query import (
    AndOperation,
    ContainsOperation,
    ContainsSpec,
    EqOperation,
    GetFieldOperator,
    GteOperation,
    GtOperation,
    LiteralOperation,
    NotOperation,
    Query,
)

from wandb_mcp_server.trace_utils import process_traces

os.environ["WANDB_SILENT"] = "True"
weave_logger = logging.getLogger("weave")
weave_logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

load_dotenv()


# Import server args - we'll use a function to avoid circular imports
def get_args():
    """Get the server arguments from the server module.

    This function avoids circular imports by importing server module only when needed.

    Returns
    -------
    Namespace  # Assuming argparse Namespace or similar
        The server arguments.
    """
    # Import here to avoid circular imports
    from wandb_mcp_server.server import get_server_args

    return get_server_args()


def get_weave_trace_server(api_key, project_id) -> weave.trace_server.trace_server_interface.TraceServerInterface:
    """Initialize and return a Weave trace server client.

    Parameters
    ----------
    api_key : str
        The W&B API key.
    project_id : str
        The Weave project ID (e.g., "entity/project").

    Returns
    -------
    weave.weave_client.WeaveClient.ServerInterface # Assuming this type
        The initialized Weave trace server interface.
    """
    weave_client = weave.init(project_id, autopatch_settings={"disable_autopatch": True})
    trace_server = weave_client.server
    return trace_server


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
) -> List[Dict]:
    """Query Weave traces with flexible filtering and sorting options.

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
            - trace_id: Filter by specific trace ID
            - call_ids: Filter by specific call IDs (string or list of strings)
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

    Returns
    -------
    List[Dict]
        List of traces matching the query parameters.

    Examples
    --------
    >>> # Get recent failed traces with costs and feedback, slower than 10s
    >>> query_traces(
    ...     entity_name="my-team",
    ...     project_name="my-project",
    ...     filters={"has_exception": True, "latency": {"$gt": 10000}},
    ...     include_costs=True,
    ...     include_feedback=True
    ... )

    >>> # Get specific columns for traces whose name contains a substring
    >>> query_traces(
    ...     entity_name="my-team",
    ...     project_name="my-project",
    ...     filters={"op_name_contains": "Evaluation.summarize"},
    ...     columns=["id", "op_name", "started_at", "summary.weave.latency_ms", "costs"]
    ... )
    """
    args = get_args()
    project_id = f"{entity_name}/{project_name}"
    trace_server = get_weave_trace_server(args.wandb_api_key, project_id)

    # Create a SortBy object
    sort = trace_server_interface.SortBy(field=sort_by, direction=sort_direction)

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

    # Create a CallsQueryReq object
    query_req = trace_server_interface.CallsQueryReq(
        project_id=project_id,
        filter=calls_filter,
        limit=limit,
        offset=offset,
        sort_by=[sort],
        query=Query(**{"$expr": query_expr}) if query_expr else None,
        include_costs=include_costs,
        include_feedback=include_feedback,
        columns=columns,
        expand_columns=expand_columns,
    )

    # Execute the query and collect results
    results = []
    try:
        # Use the non-streaming query method
        query_res = trace_server.calls_query(query_req)
        raw_calls = query_res.calls
    except Exception as e:
        logger.error(f"Error executing non-streaming query: {e}")
        # Potentially fallback or re-raise
        raise # Re-raise the exception for now

    # Process the results (similar to before, but now on the full list)
    for call in raw_calls:
        # Convert CallSchema to a dictionary for easier processing
        call_dict = {
            "id": call.id,
            "project_id": call.project_id,
            "op_name": call.op_name,
            "display_name": call.display_name,
            "trace_id": call.trace_id,
            "parent_id": call.parent_id,
            "started_at": call.started_at,
            "ended_at": call.ended_at,
            "inputs": call.inputs,
            "output": call.output,
            "exception": call.exception,
            "attributes": call.attributes,
            "summary": call.summary,
        }

        # Add costs if included
        if include_costs and hasattr(call, "costs"):
            call_dict["costs"] = call.costs

        # Add feedback if included
        if include_feedback and hasattr(call, "feedback"):
            call_dict["feedback"] = call.feedback

        # Filter the dictionary to only include requested columns if specified
        if columns:
            call_dict = {k: v for k, v in call_dict.items() if k in columns}

        results.append(call_dict)

    return results


async def paginated_query_traces(
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

    Returns
    -------
    Dict[str, Any]
        A dictionary containing 'metadata' and optionally 'traces'.
    """
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
            logger.warning(f"Invalid comparison format for {field_name}: {comparison_dict}. Expected dict with one operator key. Skipping.")
            return None

        operator, value = next(iter(comparison_dict.items()))

        try:
            field_op = GetFieldOperator(**{"$getField": field_name})
            literal_op = LiteralOperation(**{"$literal": value})
        except Exception as e:
            logger.warning(f"Invalid value for {field_name} comparison {operator}: {value}. Error: {e}. Skipping.")
            return None

        if operator == "$gt":
            return GtOperation(**{"$gt": (field_op, literal_op)})
        elif operator == "$gte":
            return GteOperation(**{"$gte": (field_op, literal_op)})
        elif operator == "$eq":
            return EqOperation(**{"$eq": (field_op, literal_op)})
        elif operator == "$lt": # Implement $lt as $not($gte)
            gte_op = GteOperation(**{"$gte": (field_op, literal_op)})
            return NotOperation(**{"$not": [gte_op]})
        elif operator == "$lte": # Implement $lte as $not($gt)
            gt_op = GtOperation(**{"$gt": (field_op, literal_op)})
            return NotOperation(**{"$not": [gt_op]})
        else:
            logger.warning(f"Unsupported comparison operator '{operator}' for {field_name}. Skipping.")
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
            comp_op = _create_comparison_op("summary.weave.status", {"$eq": target_status.lower()})
            if comp_op: operations.append(comp_op)
        else:
             logger.warning(f"Invalid status filter value: {target_status}. Expected a string. Skipping.")

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

    # Handle latency filter based on summary.weave.latency_ms
    if "latency" in filters:
        latency_filter = filters["latency"]
        comp_op = _create_comparison_op("summary.weave.latency_ms", latency_filter)
        if comp_op: operations.append(comp_op)

    # Handle attributes filter using dot notation AND supporting comparison operators
    if "attributes" in filters:
        attributes_filters = filters["attributes"]
        if isinstance(attributes_filters, dict):
            for attr_path, attr_value_or_op in attributes_filters.items():
                full_attr_path = f"attributes.{attr_path}"

                # Check if the value is a comparison operator dict or a literal
                if isinstance(attr_value_or_op, dict) and len(attr_value_or_op) == 1 and next(iter(attr_value_or_op.keys())) in ["$gt", "$gte", "$lt", "$lte", "$eq"]:
                    # It's a comparison operation
                    comp_op = _create_comparison_op(full_attr_path, attr_value_or_op)
                    if comp_op: operations.append(comp_op)
                else:
                    # Assume literal equality
                    comp_op = _create_comparison_op(full_attr_path, {"$eq": attr_value_or_op})
                    if comp_op: operations.append(comp_op)
        else:
             logger.warning(f"Invalid format for 'attributes' filter: {attributes_filters}. Expected a dictionary. Skipping.")

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
            return operations[0]
        else:
            return AndOperation(**{"$and": operations})

    return None
