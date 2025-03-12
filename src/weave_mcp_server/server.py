#!/usr/bin/env python
"""
Weave MCP Server - A Model Context Protocol server for querying Weave traces.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import weave
from datetime import datetime
from itertools import chain

from dotenv import load_dotenv
import simple_parsing
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP

# Import query_traces and our new utilities
from weave_mcp_server.query import query_traces, count_traces
from weave_mcp_server.trace_utils import process_traces

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weave-mcp-server")

# Create an MCP server using FastMCP
mcp = FastMCP("weave-mcp-server")

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Define server arguments using a dataclass for simple_parsing
@dataclass
class ServerMCPArgs:
    """Arguments for the Weave MCP Server."""

    wandb_api_key: Optional[str] = field(
        default=None, metadata=dict(help="Weights & Biases API key")
    )
    product_name: str = field(
        default="weave", metadata=dict(help="Product name (weave, wandb, or all)")
    )
    use_weave: bool = field(
        default=True, metadata=dict(help="Whether or not to trace MCP server calls to weave")
    )
    weave_entity: Optional[str] = field(
        default=None, metadata=dict(help="The Weights & Biases entity to log traced MCP server calls to")
    )
    weave_project: Optional[str] = field(
        default="weave-mcp-server", metadata=dict(help="The Weights & Biases project to log traced MCP server calls to")
    )

# Parse the command-line args
server_args = simple_parsing.parse(ServerMCPArgs)

# Get API key from environment if not provided via CLI
if not server_args.wandb_api_key:
    server_args.wandb_api_key = os.getenv("WANDB_API_KEY", "")

if server_args.use_weave:
    if server_args.weave_entity is not None:
        weave_entity_project = f"{server_args.weave_entity}/{server_args.weave_project}"
    else:
        weave_entity_project = f"{server_args.weave_project}"
    weave.init(weave_entity_project)

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
            "average_tokens_per_trace": 0
        },
        "time_range": {
            "earliest": None,
            "latest": None
        },
        "status_summary": {
            "success": 0,
            "error": 0,
            "other": 0
        },
        "op_distribution": {}
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
            if not merged["time_range"]["earliest"] or time_range["earliest"] < merged["time_range"]["earliest"]:
                merged["time_range"]["earliest"] = time_range["earliest"]
        if time_range.get("latest"):
            if not merged["time_range"]["latest"] or time_range["latest"] > merged["time_range"]["latest"]:
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
    entity_name: str,
    project_name: str,
    chunk_size: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """Query traces with pagination."""
    all_raw_traces = []
    current_offset = 0
    
    # Extract parameters for different functions
    target_limit = kwargs.pop('limit', None)
    truncate_length = kwargs.pop('truncate_length', 200)
    return_full_data = kwargs.pop('return_full_data', False)
    metadata_only = kwargs.get('metadata_only', False)
    
    # Parameters that go to query_traces
    query_params = {
        'filters': kwargs.get('filters'),
        'sort_by': kwargs.get('sort_by'),
        'sort_direction': kwargs.get('sort_direction'),
        'include_costs': kwargs.get('include_costs'),
        'include_feedback': kwargs.get('include_feedback'),
        'columns': kwargs.get('columns'),
        'expand_columns': kwargs.get('expand_columns'),
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
            **query_params
        )
        
        # Add raw traces to collection
        if not chunk_result:
            break
            
        all_raw_traces.extend(chunk_result)
        
        # Check if we should stop
        if len(chunk_result) < current_chunk_size or (target_limit and len(all_raw_traces) >= target_limit):
            break
            
        current_offset += chunk_size
    
    # Process all traces once, with appropriate parameters based on needs
    processed_result = process_traces(
        traces=all_raw_traces,
        truncate_length=truncate_length if not metadata_only else 0,  # Only truncate if we need traces
        return_full_data=return_full_data if not metadata_only else True  # Use full data for metadata
    )
    
    result = {"metadata": processed_result["metadata"]}
    
    # Add traces if needed
    if not metadata_only:
        result["traces"] = processed_result["traces"][:target_limit] if target_limit else processed_result["traces"]
        
    return result

@mcp.tool()
# @weave.op
async def query_traces_tool(
    entity_name: str,
    project_name: str,
    filters: Optional[Dict[str, Any]] = None,
    sort_by: str = "started_at",
    sort_direction: str = "desc",
    limit: int = None,
    offset: int = 0,
    include_costs: bool = True,
    include_feedback: bool = True,
    columns: Optional[List[str]] = None,
    expand_columns: Optional[List[str]] = None,
    truncate_length: Optional[int] = 200,
    return_full_data: bool = False,
    metadata_only: bool = False
) -> str:
    """
    Query Weave traces, trace metadata, and trace costs with filtering and sorting options. 
    
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
    the columns you want to return if there are certain columns you don't want to return.

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
            - trace_id: Filter by specific trace ID
            - call_ids: Filter by specific call IDs (string or list of strings). Note it is the call_id, not the
              trace id that is exposed in a weave url.
            - parent_ids: Return traces that are children of the given parent trace ids (string or list of strings)
            - status: Filter by trace status, defined as whether or not the trace had an exception or not. Can be 
                `success` or `exception`.
            - time_range: Dict with "start" and "end" datetime strings. Datetime strings should be in ISO format
                (e.g. `2024-01-01T00:00:00Z`)
            - attributes: Dict of the weave attributes of the trace.
            - has_exception: Optional[bool] to filter traces by exception status:
                - None (or key not present): Show all traces regardless of exception status
                - True: Show only traces that have exceptions (exception field is not null)
                - False: Show only traces without exceptions (exception field is null)
        sort_by: Field to sort by (started_at, ended_at, op_name, etc.). Defaults to 'started_at'
        sort_direction: Sort direction ('asc' or 'desc'). Defaults to 'desc'
        limit: Maximum number of results to return. Defaults to None
        offset: Number of results to skip (for pagination). Defaults to 0
        include_costs: Include tracked api cost information in the results. Defaults to True
        include_feedback: Include weave annotations (human labels/feedback). Defaults to True
        columns: List of specific columns to include in the results. Defaults to None (all columns).
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
                summary: typing.Optional[SummaryMap]
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
    try:
        # Use paginated query with chunks of 20
        result = await paginated_query_traces(
            entity_name=entity_name,
            project_name=project_name,
            chunk_size=50,
            filters=filters,
            sort_by=sort_by,
            sort_direction=sort_direction,
            limit=limit,
            offset=offset,
            include_costs=include_costs,
            include_feedback=include_feedback,
            columns=columns,
            expand_columns=expand_columns,
            truncate_length=truncate_length,
            return_full_data=return_full_data,
            metadata_only=metadata_only
        )
        
        return json.dumps(result, cls=DateTimeEncoder)
        
    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return f"Error querying traces: {str(e)}"

@mcp.tool()
# @weave.op
async def count_traces_tool(
    entity_name: str,
    project_name: str,
    filters: Optional[Dict[str, Any]] = None
) -> str:
    """Count Weave traces matching the given filters. 

    Returns the total number of traces in a project and the number of root 
    (i.e. "parent" or top-level) traces.

    This is more efficient than query_trace_tool when you only need the count.
    This can be useful to understand how many traces are in a project before 
    querying for them as query_trace_tool can return a lot of data.
    
    Args:
        entity_name: The Weights & Biases entity name (team or username)
        project_name: The Weights & Biases project name
        filters: Dict of filter conditions, supporting:
            - display_name: Filter by display name seen in the Weave UI (string or regex pattern)
            - op_name: Filter by weave op name, a long URI starting with 'weave:///' (string or regex pattern)
            - op_name_contains: Filter for op_name containing this substring (easier than regex)
            - trace_id: Filter by specific trace ID
            - status: Filter by trace status (success, error, etc.)
            - time_range: Dict with "start" and "end" datetime strings
            - attributes: Dict of attribute path and value to match
            - has_exception: Boolean to filter traces with/without exceptions

    Returns:
        JSON string containing the count of matching traces
    """
    try:
        # Call the synchronous count_traces function
        total_count = count_traces(
            entity_name=entity_name,
            project_name=project_name,
            filters=filters
        )

        # Create a copy of filters and ensure trace_roots_only is True
        root_filters = filters.copy() if filters else {}
        root_filters['trace_roots_only'] = True
        root_traces_count = count_traces(
            entity_name=entity_name,
            project_name=project_name,
            filters=root_filters,
        )
        
        return json.dumps({"total_count": total_count, "root_traces_count": root_traces_count})
        
    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return f"Error counting traces: {str(e)}"

def cli():
    """Command-line interface for starting the Weave MCP Server."""
    # Validate that we have the required API key
    if not server_args.wandb_api_key:
        raise ValueError("WANDB_API_KEY must be set either as an environment variable, in .env file, or as a command-line argument")
    
    print(f"Starting Weave MCP Server for {server_args.product_name}")
    logger.info(f"API Key configured: {'Yes' if server_args.wandb_api_key else 'No'}")
    
    # Run the server with stdio transport
    mcp.run(transport='stdio')

if __name__ == "__main__":
    cli()
