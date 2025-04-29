#!/usr/bin/env python
"""
Weights & Biases MCP Server - A Model Context Protocol server for querying Weights & Biases data.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from wandb_mcp_server.query_models import list_entity_projects
from wandb_mcp_server.query_weave import  paginated_query_traces
from wandb_mcp_server.tools.count_traces import count_traces, COUNT_WEAVE_TRACES_TOOL_DESCRIPTION
from wandb_mcp_server.tools.query_wandb_gql import query_paginated_wandb_gql, QUERY_WANDB_GQL_TOOL_DESCRIPTION
from wandb_mcp_server.report import create_report
from wandb_mcp_server.tool_prompts import (
    CREATE_WANDB_REPORT_TOOL_DESCRIPTION,
    LIST_ENTITY_PROJECTS_TOOL_DESCRIPTION,
    QUERY_WEAVE_TRACES_TOOL_DESCRIPTION
)
from wandb_mcp_server.query_wandbot import query_wandbot_api, WANDBOT_TOOL_DESCRIPTION
from wandb_mcp_server.trace_utils import DateTimeEncoder
from wandb_mcp_server.utils import get_server_args

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
    filters: Optional[Dict[str, Any]] = None,
    sort_by: str = "started_at",
    sort_direction: str = "desc",
    limit: int = None,
    include_costs: bool = True,
    include_feedback: bool = True,
    columns: Optional[List[str]] = None,
    expand_columns: Optional[List[str]] = None,
    truncate_length: Optional[int] = 200,
    return_full_data: bool = False,
    metadata_only: bool = False,
) -> str:
    try:
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
    entity_name: str, project_name: str, filters: Optional[Dict[str, Any]] = None
) -> str:
    try:
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
    variables: Dict[str, Any] = None,
    max_items: int = 100,
    items_per_page: int = 20,
) -> Dict[str, Any]:
    return query_paginated_wandb_gql(query, variables, max_items, items_per_page)


@mcp.tool(description=CREATE_WANDB_REPORT_TOOL_DESCRIPTION)
async def create_wandb_report_tool(
    entity_name: str,
    project_name: str,
    title: str,
    description: Optional[str] = None,
    markdown_report_text: str = None,
    plots_html: Optional[Union[Dict[str, str], str]] = None,
) -> str:
    # Handle plot_htmls if it's a JSON string
    if isinstance(plots_html, str):
        try:
            plots_html = json.loads(plots_html)
        except json.JSONDecodeError:
            # If it's not valid JSON, keep it as is (though this will likely cause other errors)
            pass

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
    wandbot_base_url = os.getenv("WANDBOT_BASE_URL")
    if not wandbot_base_url:
        raise ValueError("WANDBOT_BASE_URL environment variable is not set.")
    return query_wandbot_api(question, wandbot_base_url=wandbot_base_url)


def cli():
    """Command-line interface for starting the Weave MCP Server."""
    # Validate that we have the required API key
    if not get_server_args().wandb_api_key:
        raise ValueError(
            "WANDB_API_KEY must be set either as an environment variable, in .env file, or as a command-line argument"
        )

    print(f"Starting Weights & Biases MCP Server for {get_server_args().product_name}")
    logger.info(
        f"API Key configured: {'Yes' if get_server_args().wandb_api_key else 'No'}"
    )

    # Run the server with stdio transport
    mcp.run(transport="stdio")


if __name__ == "__main__":
    cli()
