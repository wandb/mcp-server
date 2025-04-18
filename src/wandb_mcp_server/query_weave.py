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
    GtOperation,
    LiteralOperation,
    NotOperation,
    Query,
)

import wandb
import weave
from wandb_mcp_server.trace_utils import process_traces

logger = logging.getLogger(__name__)

load_dotenv()


# Import server args - we'll use a function to avoid circular imports
def get_args():
    """Get the server arguments from the server module.

    This function avoids circular imports by importing server module only when needed.
    """
    # Import here to avoid circular imports
    from wandb_mcp_server.server import get_server_args

    return get_server_args()


def get_weave_trace_server(api_key, project_id):
    wandb.login(key=api_key)
    weave_client = weave.init(project_id)
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
    """
    Query Weave traces with flexible filtering and sorting options. Always ensure that the entity_name and
    project_name have been provided by the user.

    Args:
        entity_name: The Weights & Biases entity name. This can be a W&B team or username.
        project_name: The Weights & Biases project name
        filters: Dict of filter conditions, supporting:
            - display_name: Filter by display name seen in the Weave UI (string or regex pattern)
            - op_name: Filter by weave op name, a long URI starting with 'weave:///'  (string or regex pattern)
            - op_name_contains: Filter for op_name containing this substring (easier than regex)
            - trace_id: Filter by specific trace ID
            - call_ids: Filter by specific call IDs (string or list of strings)
            - status: Filter by trace status, can be `success`, `error`, or `running`
            - time_range: Dict with "start" and "end" datetime strings
            - attributes: Dict of attribute path and value to match
            - has_exception: Optional[bool] to filter traces by exception status:
                - None (or key not present): Show all traces regardless of exception status
                - True: Show only traces that have exceptions (exception field is not null)
                - False: Show only traces without exceptions (exception field is null)
            - trace_roots_only: Boolean to filter for only top-level/parent traces
        sort_by: Field to sort by (started_at, ended_at, op_name, etc.)
        sort_direction: Sort direction ("asc" or "desc")
        limit: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        include_costs: Include tracked api cost information in the results
        include_feedback: Include weave annotataions (for example human labels or feedback) information in the results
        columns: List of specific columns to include in the results
        expand_columns: List of columns to expand in the results

    Returns:
        List of traces matching the query parameters

    Example:
        ```python
        # Get recent failed traces with costs and feedback
        query_traces(
            entity_name="my-team",
            project_name="my-project",
            filters={"has_exception": True},  # Only show traces with exceptions
            include_costs=True,
            include_feedback=True
        )

        # Get specific columns for traces who's name contains a substring
        query_traces(
            entity_name="my-team",
            project_name="my-project",
            filters={"op_name": "Evaluation.summarize"},
            columns=["id", "op_name", "started_at", "costs"]
        )
        ```
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
    for call in trace_server.calls_query_stream(query_req):
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
    entity_name: str, project_name: str, chunk_size: int = 20, **kwargs
) -> Dict[str, Any]:
    """Query traces with pagination."""
    all_raw_traces = []
    current_offset = 0

    # Extract parameters for different functions
    target_limit = kwargs.pop("limit", None)
    truncate_length = kwargs.pop("truncate_length", 200)
    return_full_data = kwargs.pop("return_full_data", False)
    metadata_only = kwargs.get("metadata_only", False)

    # Parameters that go to query_traces
    query_params = {
        "filters": kwargs.get("filters"),
        "sort_by": kwargs.get("sort_by"),
        "sort_direction": kwargs.get("sort_direction"),
        "include_costs": kwargs.get("include_costs"),
        "include_feedback": kwargs.get("include_feedback"),
        "columns": kwargs.get("columns"),
        "expand_columns": kwargs.get("expand_columns"),
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
    """Convert an ISO format datetime string to Unix timestamp."""
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
    """
    Build a Query expression from the filter dictionary.

    Args:
        filters: Dictionary of filter conditions

    Returns:
        Query object or None if no valid filters are provided
    """
    operations = []

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

    # Handle status filter
    if "status" in filters:
        status = filters["status"]
        # Convert status to has_exception
        if status.lower() == "exception":
            filters["has_exception"] = True
        elif status.lower() == "success":
            filters["has_exception"] = False

    # Handle has_exception filter
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

    # Handle time range filter
    if "time_range" in filters:
        time_range = filters["time_range"]

        if "start" in time_range:
            start_timestamp = _datetime_to_timestamp(time_range["start"])
            if start_timestamp > 0:
                operations.append(
                    GtOperation(
                        **{
                            "$gt": (
                                GetFieldOperator(**{"$getField": "started_at"}),
                                LiteralOperation(**{"$literal": start_timestamp}),
                            )
                        }
                    )
                )

        if "end" in time_range:
            end_timestamp = _datetime_to_timestamp(time_range["end"])
            if end_timestamp > 0:
                operations.append(
                    GtOperation(
                        **{
                            "$gt": (
                                LiteralOperation(**{"$literal": end_timestamp}),
                                GetFieldOperator(**{"$getField": "started_at"}),
                            )
                        }
                    )
                )

    # Handle attributes filter
    if "attributes" in filters:
        attributes = filters["attributes"]
        for attr_path, attr_value in attributes.items():
            # Create a path to the attribute
            attr_path_parts = attr_path.split(".")

            # Build the nested GetFieldOperator for the attribute path
            field_op = GetFieldOperator(**{"$getField": "attributes"})
            for part in attr_path_parts:
                field_op = GetFieldOperator(**{"$getField": part, "input": field_op})

            # Add the equality operation
            operations.append(
                EqOperation(
                    **{"$eq": (field_op, LiteralOperation(**{"$literal": attr_value}))}
                )
            )

    # Combine all operations with AND
    if operations:
        if len(operations) == 1:
            return operations[0]
        else:
            return AndOperation(**{"$and": operations})

    return None


def count_traces(
    entity_name: str, project_name: str, filters: Dict[str, Any] = None
) -> int:
    """
    Count the number of traces matching the given filters without retrieving the full trace data.
    This is more efficient than query_traces when you only need the count.

    Args:
        entity_name: The Weights & Biases entity name. This can be a W&B team or username.
        project_name: The Weights & Biases project name
        filters: Dict of filter conditions, supporting:
            - display_name: Filter by display name seen in the Weave UI (string or regex pattern)
            - op_name: Filter by weave op name, a long URI starting with 'weave:///'  (string or regex pattern)
            - op_name_contains: Filter for op_name containing this substring (easier than regex)
            - trace_id: Filter by specific trace ID
            - status: Filter by trace status (success, error, etc.)
            - time_range: Dict with "start" and "end" datetime strings
            - attributes: Dict of attribute path and value to match
            - has_exception: Boolean to filter traces with/without exceptions
            - trace_roots_only: Boolean to filter for only top-level/parent traces

    Returns:
        Integer count of traces matching the query parameters

    Example:
        ```python
        # Count failed traces
        count = count_traces(
            entity_name="my-team",
            project_name="my-project",
            filters={"status": "error"}
        )
        ```
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
    stats_req = trace_server_interface.CallsQueryStatsReq(
        project_id=project_id,
        filter=calls_filter,
        query=Query(**{"$expr": query_expr}) if query_expr else None,
    )

    # Execute the query and get the count
    stats_res = trace_server.calls_query_stats(stats_req)
    return stats_res.count
