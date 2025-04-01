from typing import Any, Dict, List
import logging

from mcp_server.query_weave import query_traces
from mcp_server.trace_utils import process_traces

logger = logging.getLogger(__name__)


def merge_metadata(metadata_list: List[Dict]) -> Dict:
    """Merge metadata from multiple query results."""
    if not metadata_list:
        return {}

    merged = {
        "total_traces": 0,
        "token_counts": {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "average_tokens_per_trace": 0,
        },
        "time_range": {"earliest": None, "latest": None},
        "status_summary": {"success": 0, "error": 0, "other": 0},
        "op_distribution": {},
    }

    for metadata in metadata_list:
        # Sum up trace counts
        merged["total_traces"] += metadata.get("total_traces", 0)

        # Sum up token counts
        token_counts = metadata.get("token_counts", {})
        merged["token_counts"]["total_tokens"] += token_counts.get("total_tokens", 0)
        merged["token_counts"]["input_tokens"] += token_counts.get("input_tokens", 0)
        merged["token_counts"]["output_tokens"] += token_counts.get("output_tokens", 0)

        # Update time range
        time_range = metadata.get("time_range", {})
        if time_range.get("earliest"):
            if (
                not merged["time_range"]["earliest"]
                or time_range["earliest"] < merged["time_range"]["earliest"]
            ):
                merged["time_range"]["earliest"] = time_range["earliest"]
        if time_range.get("latest"):
            if (
                not merged["time_range"]["latest"]
                or time_range["latest"] > merged["time_range"]["latest"]
            ):
                merged["time_range"]["latest"] = time_range["latest"]

        # Sum up status counts
        status_summary = metadata.get("status_summary", {})
        merged["status_summary"]["success"] += status_summary.get("success", 0)
        merged["status_summary"]["error"] += status_summary.get("error", 0)
        merged["status_summary"]["other"] += status_summary.get("other", 0)

        # Merge op distributions
        for op, count in metadata.get("op_distribution", {}).items():
            merged["op_distribution"][op] = merged["op_distribution"].get(op, 0) + count

    # Calculate average tokens per trace
    if merged["total_traces"] > 0:
        merged["token_counts"]["average_tokens_per_trace"] = (
            merged["token_counts"]["total_tokens"] / merged["total_traces"]
        )

    return merged


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