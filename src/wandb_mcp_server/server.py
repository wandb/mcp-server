#!/usr/bin/env python
"""
Weights & Biases MCP Server - A Model Context Protocol server for querying Weights & Biases data.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import sys # Added for stdout redirection
import io # Added for stdout redirection

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import wandb # Added for wandb.login and wandb.setup

from wandb_mcp_server.query_models import list_entity_projects
from wandb_mcp_server.query_weave import  paginated_query_traces
from wandb_mcp_server.tools.count_traces import count_traces, COUNT_WEAVE_TRACES_TOOL_DESCRIPTION
from wandb_mcp_server.tools.query_wandb_gql import query_paginated_wandb_gql, QUERY_WANDB_GQL_TOOL_DESCRIPTION
from wandb_mcp_server.tools.query_wandbot import query_wandbot_api, WANDBOT_TOOL_DESCRIPTION
from wandb_mcp_server.report import create_report
from wandb_mcp_server.tool_prompts import (
    CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
    LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION,
    QUERY_WEAVE_TRACES_TOOL_DESCRIPTION
)
from wandb_mcp_server.trace_utils import DateTimeEncoder
from wandb_mcp_server.utils import get_server_args

# Silence logging to avoid interfering with MCP server
os.environ["WANDB_SILENT"] = "True"
weave_logger = logging.getLogger("weave")
weave_logger.setLevel(logging.ERROR)
gql_transport_logger = logging.getLogger("gql.transport.requests")
gql_transport_logger.setLevel(logging.ERROR)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weave-mcp-server")

# Create an MCP server using FastMCP
mcp = FastMCP("weave-mcp-server")


@mcp.tool(description=QUERY_WEAVE_TRACES_TOOL_DESCRIPTION)
async def query_weave_traces_tool(
    entity_name: str,
    project_name: str,
    filters_json: Optional[str] = None,
    sort_by: str = "started_at",
    sort_direction: str = "desc",
    limit: Optional[int] = None,
    include_costs: bool = True,
    include_feedback: bool = True,
    columns_json: Optional[str] = None,
    expand_columns_json: Optional[str] = None,
    truncate_length: Optional[int] = 200,
    return_full_data: bool = False,
    metadata_only: bool = False,
) -> str:
    """Query traces from W&B Weave.
    
    Args:
        entity_name: The W&B entity/username
        project_name: The W&B project name
        filters_json: Optional JSON string of filters to apply (e.g., '{"trace_roots_only": true}')
        sort_by: Field to sort by (default: "started_at")
        sort_direction: Sort direction, "asc" or "desc" (default: "desc")
        limit: Maximum number of traces to return
        include_costs: Whether to include cost information (default: true)
        include_feedback: Whether to include feedback information (default: true)
        columns_json: Optional JSON string array of columns to include (e.g., '["input", "output"]')
        expand_columns_json: Optional JSON string array of columns to expand (e.g., '["input", "output"]')
        truncate_length: Maximum length for truncated fields (default: 200)
        return_full_data: Whether to return full data (default: false)
        metadata_only: Whether to return only metadata (default: false)
    
    Returns:
        JSON string with query results
    """
    try:
        # Parse JSON strings into Python objects
        filters = json.loads(filters_json) if filters_json else None
        columns = json.loads(columns_json) if columns_json else None
        expand_columns = json.loads(expand_columns_json) if expand_columns_json else None
        
        # Use paginated query with chunks of 20
        result = await paginated_query_traces(
            entity_name=entity_name,
            project_name=project_name,
            chunk_size=50,
            filters=filters,
            sort_by=sort_by,
            sort_direction=sort_direction,
            target_limit=limit,
            include_costs=include_costs,
            include_feedback=include_feedback,
            columns=columns,
            expand_columns=expand_columns,
            truncate_length=truncate_length,
            return_full_data=return_full_data,
            metadata_only=metadata_only,
        )

        return json.dumps(result, cls=DateTimeEncoder)

    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return f"Error querying traces: {str(e)}"


@mcp.tool(description=COUNT_WEAVE_TRACES_TOOL_DESCRIPTION)
async def count_weave_traces_tool(
    entity_name: str, project_name: str, filters_json: Optional[str] = None
) -> str:
    """Count traces in a W&B Weave project.
    
    Args:
        entity_name: The W&B entity/username
        project_name: The W&B project name
        filters_json: Optional JSON string of filters to apply (e.g., '{"trace_roots_only": true}')
    
    Returns:
        JSON string with total count and root traces count
    """
    try:
        # Parse JSON string into Python object
        filters = json.loads(filters_json) if filters_json else None
        
        # Call the synchronous count_traces function
        total_count = count_traces(
            entity_name=entity_name, project_name=project_name, filters=filters
        )

        # Create a copy of filters and ensure trace_roots_only is True
        root_filters = filters.copy() if filters else {}
        root_filters["trace_roots_only"] = True
        root_traces_count = count_traces(
            entity_name=entity_name,
            project_name=project_name,
            filters=root_filters,
        )

        return json.dumps(
            {"total_count": total_count, "root_traces_count": root_traces_count}
        )

    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return f"Error counting traces: {str(e)}"


@mcp.tool(description=QUERY_WANDB_GQL_TOOL_DESCRIPTION)
def query_wandb_gql_tool(
    query: str,
    variables_json: Optional[str] = None,
    max_items: int = 100,
    items_per_page: int = 20,
) -> Dict[str, Any]:
    """Query W&B GraphQL API with pagination support.
    
    Args:
        query: The GraphQL query string
        variables_json: Optional JSON string of variables for the query (e.g., '{"entityName": "my-entity"}')
        max_items: Maximum number of items to return (default: 100)
        items_per_page: Number of items to fetch per page (default: 20)
    
    Returns:
        Dictionary with query results
    """
    # Parse JSON string into Python object
    variables = json.loads(variables_json) if variables_json else None
    
    return query_paginated_wandb_gql(query, variables, max_items, items_per_page)


@mcp.tool(description=CREATE_WANDB_REPORT_TOOL_DESCRIPTION)
async def create_wandb_report_tool(
    entity_name: str,
    project_name: str,
    title: str,
    description: Optional[str] = None,
    markdown_report_text: Optional[str] = None,
    plots_html_json: Optional[str] = None,
) -> str:
    """Create a W&B report with markdown text and optional plots.
    
    Args:
        entity_name: The W&B entity/username
        project_name: The W&B project name
        title: The title of the report
        description: Optional description for the report
        markdown_report_text: Optional markdown text for the report content
        plots_html_json: Optional JSON string of plot HTML content (e.g., '{"plot1": "<div>...</div>"}')
    
    Returns:
        String with the URL to the created report
    """
    # Parse plots_html from JSON string
    plots_html = None
    if plots_html_json:
        try:
            plots_html = json.loads(plots_html_json)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in plots_html_json: {plots_html_json}")
            return "Error: Invalid JSON format for plots_html_json"

    report_link = create_report(
        entity_name=entity_name,
        project_name=project_name,
        title=title,
        description=description,
        markdown_report_text=markdown_report_text,
        plots_html=plots_html,
    )
    return f"The report was saved here: {report_link}"


@mcp.tool(description=LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION)
def query_wandb_entity_projects(entity: Optional[str] = None) -> List[Dict[str, Any]]:
    return list_entity_projects(entity)


@mcp.tool(description=WANDBOT_TOOL_DESCRIPTION)
def query_wandb_support_bot(question: str) -> str:
    return query_wandbot_api(question)


def cli():
    """Command-line interface for starting the Weave MCP Server."""
    # Ensure WANDB_SILENT is set, and attempt to configure wandb for silent operation globally
    os.environ["WANDB_SILENT"] = "True"
    try:
        wandb.setup(settings=wandb.Settings(silent=True, console="off"))
    except Exception as e:
        logger.warning(f"Could not apply wandb.setup settings: {e}")

    # Attempt to explicitly login to W&B and suppress its stdout messages
    # This is to ensure login happens before mcp.run() and to capture login confirmations.
    api_key = get_server_args().wandb_api_key
    if api_key:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = captured_stdout = io.StringIO()
        sys.stderr = captured_stderr = io.StringIO()
        try:
            logger.info("Attempting explicit W&B login in cli()...")
            wandb.login(key=api_key)
            login_msg_stdout = captured_stdout.getvalue().strip()
            login_msg_stderr = captured_stderr.getvalue().strip()
            if login_msg_stdout:
                logger.info(f"Suppressed stdout during W&B login: {login_msg_stdout}")
            if login_msg_stderr:
                logger.info(f"Suppressed stderr during W&B login: {login_msg_stderr}")
            logger.info("Explicit W&B login attempt finished.")
        except Exception as e:
            logger.error(f"Error during explicit W&B login: {e}")
            # Potentially re-raise or handle as a fatal error if login is critical
        finally:
            sys.stdout = original_stdout # Always restore stdout
            sys.stderr = original_stderr # Always restore stderr
    else:
        logger.warning("WANDB_API_KEY not found via get_server_args(). Skipping explicit login.")

    # Validate that we have the required API key (may be redundant if explicit login was attempted)
    if not get_server_args().wandb_api_key: # Re-check, as get_server_args might have complex logic or state
        raise ValueError(
            "WANDB_API_KEY must be set either as an environment variable, in .env file, or as a command-line argument"
        )

    logger.info(f"Starting Weights & Biases MCP Server.")
    logger.info(
        f"API Key configured: {'Yes' if get_server_args().wandb_api_key else 'No'}"
    )

    # Run the server with stdio transport
    mcp.run(transport="stdio")


if __name__ == "__main__":
    cli()
