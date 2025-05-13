"""
Raw HTTP-based implementation of the Weave API client.
This module provides the same functionality as the original Weave client-based implementation
but uses raw HTTP requests to interact with the Weave server.
"""

import os
import calendar
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import json
from dotenv import load_dotenv

# Import the query models for building complex queries
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
    ConvertOperation,
)

from wandb_mcp_server.trace_utils import process_traces
from wandb_mcp_server.utils import RedirectLoggerHandler

os.environ["WANDB_SILENT"] = "True"
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
    from wandb_mcp_server.utils import get_server_args

    return get_server_args()


def get_weave_server_url() -> str:
    """Get the Weave server URL.
    
    Returns
    -------
    str
        The Weave server URL.
    """
    # Default Weave server URL
    return os.environ.get("WEAVE_TRACE_SERVER_URL", "https://trace.wandb.ai")


def get_weave_trace_server(api_key, project_id):
    """Initialize and return a Weave Trace Server Interface.
    
    This is a compatibility function that doesn't actually return a TraceServerInterface,
    but instead returns a dummy object that can be used with the raw HTTP implementation.

    Parameters
    ----------
    api_key : str
        The W&B API key.
    project_id : str
        The Weave project ID (e.g., "entity/project").

    Returns
    -------
    object
        A dummy object that can be used with the raw HTTP implementation.
    """
    # Configure weave logger to redirect to the main logger of this module
    weave_logger_instance = logging.getLogger("weave")
    weave_logger_instance.propagate = False 

    for handler in weave_logger_instance.handlers[:]:
        weave_logger_instance.removeHandler(handler)

    redirect_handler = RedirectLoggerHandler(logger) 
    weave_logger_instance.addHandler(redirect_handler)
    
    # Set the API key in the environment
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    
    # Return a dummy object that can be used with the raw HTTP implementation
    class DummyTraceServer:
        def __init__(self, api_key, project_id):
            self.api_key = api_key
            self.project_id = project_id
        
        def calls_query(self, req):
            # Extract parameters from the request
            entity_name, project_name = self.project_id.split("/")
            
            # Convert the request to parameters for query_traces
            filters = {}
            if req.filter:
                # Convert CallsFilter to dict
                filter_dict = req.filter.model_dump()
                for key, value in filter_dict.items():
                    if value is not None:
                        filters[key] = value
            
            # Convert Query to dict
            query_expr = None
            if req.query:
                query_expr = req.query
            
            # Convert sort_by to parameters
            sort_by = None
            sort_direction = "desc"
            if req.sort_by and len(req.sort_by) > 0:
                sort_by = req.sort_by[0].field
                sort_direction = req.sort_by[0].direction
            
            # Call query_traces
            traces = query_traces(
                entity_name=entity_name,
                project_name=project_name,
                filters=filters,
                sort_by=sort_by,
                sort_direction=sort_direction,
                limit=req.limit,
                offset=req.offset,
                include_costs=req.include_costs,
                include_feedback=req.include_feedback,
                columns=req.columns,
                expand_columns=req.expand_columns,
                api_key=self.api_key,
                query_expr=query_expr,
            )
            
            # Return a response object with the traces
            class CallsQueryRes:
                def __init__(self, calls):
                    self.calls = calls
            
            return CallsQueryRes(calls=traces)
    
    return DummyTraceServer(api_key, project_id)


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
    api_key : Optional[str], optional
        The W&B API key. If not provided, will try to get from environment.
    query_expr : Optional[Query], optional
        A pre-built Query object to use instead of building one from filters.

    Returns
    -------
    List[Dict]
        List of traces matching the query parameters.
    """
    if api_key is None:
        # Get API key from server args
        args = get_args()
        api_key = args.wandb_api_key
    
    project_id = f"{entity_name}/{project_name}"
    
    # --- Filter Separation ---
    direct_filters_for_callsfilter_model = {}
    complex_filters_for_expr = {}

    if filters:
        # Simple filters for CallsFilter object
        simple_filter_keys = ["trace_roots_only", "op_names", "trace_ids", "parent_ids", "call_ids"]
        for key in simple_filter_keys:
            if key in filters:
                # Ensure op_names, trace_ids, etc. are lists as expected by CallsFilter
                if key in ["op_names", "trace_ids", "parent_ids", "call_ids"] and not isinstance(filters[key], list):
                    direct_filters_for_callsfilter_model[key] = [str(filters[key])]
                else:
                    direct_filters_for_callsfilter_model[key] = filters[key]

        # Handle individual op_name and trace_id if op_names/trace_ids not already set
        if "op_name" in filters and "op_names" not in direct_filters_for_callsfilter_model:
            # Only add if it's a simple name, not a pattern (patterns go to complex)
            if isinstance(filters["op_name"], str) and "*" not in filters["op_name"] and ".*" not in filters["op_name"]:
                direct_filters_for_callsfilter_model["op_names"] = [filters["op_name"]]
            else:
                # It's a pattern or complex, send to complex_filters_for_expr
                complex_filters_for_expr["op_name"] = filters["op_name"]
        elif "op_name" in filters and "op_names" in direct_filters_for_callsfilter_model:
             # If op_names is already set, and op_name is a pattern, it needs to go to complex
            if isinstance(filters["op_name"], str) and ("*" in filters["op_name"] or ".*" in filters["op_name"]):
                 complex_filters_for_expr["op_name"] = filters["op_name"]

        if "trace_id" in filters and "trace_ids" not in direct_filters_for_callsfilter_model:
            direct_filters_for_callsfilter_model["trace_ids"] = [str(filters["trace_id"])]

        # Complex filters for $expr via _build_query_expression
        # All other keys from the original `filters` dict go here.
        all_handled_direct_keys = set(direct_filters_for_callsfilter_model.keys())
        # Add op_name/trace_id to handled if they were processed into op_names/trace_ids
        if "op_names" in direct_filters_for_callsfilter_model and "op_name" in filters and filters["op_name"] in direct_filters_for_callsfilter_model["op_names"]:
            all_handled_direct_keys.add("op_name")
        if "trace_ids" in direct_filters_for_callsfilter_model and "trace_id" in filters and filters["trace_id"] in direct_filters_for_callsfilter_model["trace_ids"]:
            all_handled_direct_keys.add("trace_id")
            
        for key, value in filters.items():
            if key not in all_handled_direct_keys:
                 # Exception: if op_name was simple and handled, but is also in filters, it shouldn't be added again
                if key == "op_name" and "op_names" in direct_filters_for_callsfilter_model and value in direct_filters_for_callsfilter_model["op_names"] and not ("*" in value or ".*" in value):
                    continue
                complex_filters_for_expr[key] = value
    
    # Build the request body
    request_body = {
        "project_id": project_id,
    }
    
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
    if columns:
        request_body["columns"] = columns
    if expand_columns:
        request_body["expand_columns"] = expand_columns
    
    # Make the HTTP request
    url = f"{get_weave_server_url()}/calls/stream_query"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/jsonl"
    }
    
    # Set up authentication
    auth = None
    if api_key:
        auth = (api_key, "")
    else:
        # Try to get API key from environment
        env_api_key = os.environ.get("WANDB_API_KEY")
        if env_api_key:
            auth = (env_api_key, "")
    
    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(request_body),
            auth=auth,
            stream=True
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
                call_data = json.loads(line.decode('utf-8'))
                
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
                
                # Add costs if included
                if include_costs and "costs" in call_data:
                    call_dict["costs"] = call_data.get("costs")
                
                # Add feedback if included
                if include_feedback and "feedback" in call_data:
                    call_dict["feedback"] = call_data.get("feedback")
                
                # Column filtering should have been handled by the server if `columns` was passed
                if columns:
                    call_dict = {k: v for k, v in call_dict.items() if k in columns}
                
                results.append(call_dict)
        
        return results
    
    except Exception as e:
        logger.error(f"Error executing HTTP request to Weave server: {e}")
        raise


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
        # Get API key from server args
        args = get_args()
        api_key = args.wandb_api_key
        
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
            logger.warning(f"Invalid comparison format for {field_name}: {comparison_dict}. Expected dict with one operator key. Skipping.")
            return None

        operator, value = next(iter(comparison_dict.items()))

        try:
            field_op_base = GetFieldOperator(**{"$getField": field_name})
            # Determine if conversion is needed based on field_name pattern
            # Add more patterns as needed (e.g., specific known numeric attributes)
            # Re-enable conversion logic
            field_op = field_op_base # Default to base
            if field_name.startswith("attributes.") or \
               field_name == "summary.weave.latency_ms":
                # For these, if the comparison value is numeric, convert field to double.
                if isinstance(value, (int, float)):
                    field_op = ConvertOperation(
                        **{
                            "$convert": {
                                "input": field_op_base,
                                "to": "double"
                            }
                        }
                    )
                # else: field_op remains field_op_base if value is not numeric
            # Timestamps (started_at, ended_at) should NOT be converted to double here.
            # They will be compared against numeric Unix timestamps directly.
            elif field_name == "started_at" or field_name == "ended_at":
                field_op = field_op_base 
            # else: field_op remains field_op_base for other field types
            
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

    # Handle wb_run_id filter (top-level)
    if "wb_run_id" in filters:
        run_id = filters["wb_run_id"]
        # This filter expects a string for wb_run_id and uses $contains or $eq.
        # It should not be converted to double.
        if isinstance(run_id, str):
            if "$contains" in run_id or "*" in run_id : # Simple check for contains style
                pattern = run_id.replace("$contains:", "").replace("*","") # Basic cleanup
                operations.append(
                    ContainsOperation(
                        **{
                            "$contains": ContainsSpec(
                                input=GetFieldOperator(**{"$getField": "wb_run_id"}),
                                substr=LiteralOperation(**{"$literal": pattern.strip()}),
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
        elif isinstance(run_id, dict) and "$contains" in run_id: # wb_run_id: {"$contains": "foo"}
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
                logger.warning(f"Invalid $contains value for wb_run_id: {pattern}. Expected string.")
        else:
            logger.warning(f"Invalid wb_run_id filter value: {run_id}. Expected a string or dict with $contains. Skipping.")

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

                # Check for $contains operation in attributes
                if isinstance(attr_value_or_op, dict) and "$contains" in attr_value_or_op:
                    if isinstance(attr_value_or_op["$contains"], str):
                        contains_op = ContainsOperation(
                            **{
                                "$contains": ContainsSpec(
                                    input=GetFieldOperator(**{"$getField": full_attr_path}),
                                    substr=LiteralOperation(**{"$literal": attr_value_or_op["$contains"]}),
                                    case_insensitive=True,
                                )
                            }
                        )
                        operations.append(contains_op)
                    else:
                        logger.warning(f"Invalid value for $contains on {full_attr_path}: {attr_value_or_op['$contains']}. Expected string.")
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
            # Wrap the single operation in the Query model structure
            return Query(**{"$expr": operations[0]}) 
        else:
            # Wrap the AndOperation in the Query model structure
            and_op = AndOperation(**{"$and": operations})
            return Query(**{"$expr": and_op})

    return None # No complex filters, so no Query object needed for the 'query' arg