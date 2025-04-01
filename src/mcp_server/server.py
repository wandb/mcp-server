#!/usr/bin/env python
"""
Weave MCP Server - A Model Context Protocol server for querying Weave traces.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import simple_parsing
import wandb_workspaces.reports.v2 as wr
import weave
import wandb
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Import query_traces and our new utilities
from mcp_server.query_weave import count_traces, query_traces
from mcp_server.report import create_report
from mcp_server.trace_utils import process_traces
from mcp_server.utils import paginated_query_traces

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
        default=True,
        metadata=dict(help="Whether or not to trace MCP server calls to weave"),
    )
    weave_entity: Optional[str] = field(
        default=None,
        metadata=dict(
            help="The Weights & Biases entity to log traced MCP server calls to"
        ),
    )
    weave_project: Optional[str] = field(
        default="weave-mcp-server",
        metadata=dict(
            help="The Weights & Biases project to log traced MCP server calls to"
        ),
    )


# Global variable to store server args
_server_args = None


def get_server_args():
    """Get the server arguments, parsing them if not already done."""
    global _server_args
    if _server_args is None:
        _server_args = ServerMCPArgs()
        # Only parse args when explicitly requested, not at import time
        if os.environ.get("PARSE_ARGS_AT_IMPORT", "0") == "1":
            _server_args = simple_parsing.parse(ServerMCPArgs)

        # Get API key from environment if not provided via CLI
        if not _server_args.wandb_api_key:
            _server_args.wandb_api_key = os.getenv("WANDB_API_KEY", "")

    return _server_args


if get_server_args().use_weave:
    if get_server_args().weave_entity is not None:
        weave_entity_project = (
            f"{get_server_args().weave_entity}/{get_server_args().weave_project}"
        )
    else:
        weave_entity_project = f"{get_server_args().weave_project}"
    weave.init(weave_entity_project)


@mcp.tool()
async def create_wandb_report_tool(
    entity_name: str,
    project_name: str,
    title: str,
    description: Optional[str] = None,
    markdown_report_text: str = None,
    plots_html: Optional[Union[Dict[str, str], str]] = None,
) -> wr.Report:
    """
    Create a new Weights & Biases Report and add text and HTML-rendered charts. Useful to save/document analysis and other findings.

    Always provide the returned report link to the user.

    <plots_html_usage_guide>
    If the analsis has generated plots then they can be logged to a Weights & Biases report via converting them to html.
    All charts should be properly rendered in raw HTML, do not use any placeholders for any chart, render everything.
    Plot html code should use SVG chart elements that should render properly in any modern browser.
    Include interactive hover effects where it makes sense.
    If the analysis contains multiple charts, break up the html into one section of html per chart.
    </plots_html_usage_guide>

    Args:
        entity_name: str, The W&B entity (team or username) - required
        project_name: str, The W&B project name - required
        title: str, Title of the W&B Report - required
        description: str, Optional description of the W&B Report
        markdown_report_text: str, beuatifully formatted markdown text for the report body
        plots_html: str, Optional dict of plot name and html string of any charts created as part of an analysis

    Returns:
        str, The url to the report

    Example:
        ```python
        # Create a simple report
        report = create_report(
            entity_name="my-team",
            project_name="my-project",
            title="Model Analysis Report",
            description="Analysis of our latest model performance",
            markdown_report_text='''
                # Model Analysis Report
                [TOC]
                ## Performance Summary
                Our model achieved 95% accuracy on the test set.
                ### Key Metrics
                Precision: 0.92
                Recall: 0.89
            '''
        )
        ```
    """
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


@mcp.tool()
# @weave.op
async def query_weave_traces_tool(
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
    metadata_only: bool = False,
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
            metadata_only=metadata_only,
        )

        return json.dumps(result, cls=DateTimeEncoder)

    except Exception as e:
        logger.error(f"Error calling tool: {e}")
        return f"Error querying traces: {str(e)}"


@mcp.tool()
# @weave.op
async def count_weave_traces_tool(
    entity_name: str, project_name: str, filters: Optional[Dict[str, Any]] = None
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



@mcp.tool()
def query_wandb_graphql(query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute an arbitrary GraphQL query against the wandb API.
    
    Args:
        query (str): The GraphQL query string
        variables (Dict[str, Any], optional): Variables to pass to the query
        
    Returns:
        Dict[str, Any]: The query result
        
    Example:
        query = '''
        query Project($entity: String!, $name: String!) {
            project(entityName: $entity, name: $name) {
                name
                entity
                description
                runs {
                    edges {
                        node {
                            id
                            name
                            state
                        }
                    }
                }
            }
        }
        '''
        variables = {
            "entity": "my-entity",
            "name": "my-project"
        }
        result = query_wandb_graphql(query, variables)
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Execute the query
    result = api.client.execute(query, variables or {})
    
    return result


@mcp.tool()
def query_wandb_projects(entity: str) -> List[Dict[str, Any]]:
    """
    Fetch all projects for a specific wandb entity.
    
    Args:
        entity (str): The wandb entity (username or team name)
        
    Returns:
        List[Dict[str, Any]]: List of project dictionaries containing:
            - name: Project name
            - entity: Entity name
            - description: Project description
            - visibility: Project visibility (public/private)
            - created_at: Creation timestamp
            - updated_at: Last update timestamp
            - tags: List of project tags
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Get all projects for the entity
    projects = api.projects(entity)
    
    # Convert projects to a list of dictionaries
    projects_data = []
    for project in projects:
        project_dict = {
            "name": project.name,
            "entity": project.entity,
            "description": project.description,
            "visibility": project.visibility,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "tags": project.tags,
        }
        projects_data.append(project_dict)
    
    return projects_data

@mcp.tool()
def query_wandb_runs(
    entity: str,
    project: str,
    per_page: int = 50,
    order: str = "-created_at",
    filters: Dict[str, Any] = None,
    search: str = None
) -> List[Dict[str, Any]]:
    """
    Fetch runs from a specific wandb entity and project with filtering and sorting support.
    
    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        per_page (int): Number of runs to fetch (default: 50)
        order (str): Sort order (default: "-created_at"). Prefix with "-" for descending order.
                    Examples: "created_at", "-created_at", "name", "-name", "state", "-state"
        filters (Dict[str, Any]): Dictionary of filters to apply. Keys can be:
            - state: "running", "finished", "crashed", "failed", "killed"
            - tags: List of tags to filter by
            - config: Dictionary of config parameters to filter by
            - summary: Dictionary of summary metrics to filter by
        search (str): Search string to filter runs by name or tags
        
    Returns:
        List[Dict[str, Any]]: List of run dictionaries containing run information
    """
    # Initialize wandb API
    api = wandb.Api()
    
    # Build query parameters
    query_params = {
        "per_page": per_page,
        "order": order
    }
    
    # Add filters if provided
    if filters:
        for key, value in filters.items():
            if key in ["state", "tags", "config", "summary"]:
                query_params[key] = value
    
    # Add search if provided
    if search:
        query_params["search"] = search
    
    # Get runs from the specified entity and project with filters
    runs = api.runs(
        f"{entity}/{project}",
        **query_params
    )
    
    # Convert runs to a list of dictionaries
    runs_data = []
    for run in runs:
        run_dict = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "config": run.config,
            "summary": run.summary,
            "created_at": run.created_at,
            "url": run.url,
            "tags": run.tags,
        }
        runs_data.append(run_dict)
    
    return runs_data


@mcp.tool()
def query_wandb_run_config(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch configuration parameters for a specific run.
    
    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        run_id (str): The ID of the run to fetch config for
        
    Returns:
        Dict[str, Any]: Dictionary containing configuration parameters
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return run.config


@mcp.tool()
def query_wandb_run_training_metrics(entity: str, project: str, run_id: str) -> Dict[str, List[Any]]:
    """
    Fetch training metrics history for a specific run.
    
    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        run_id (str): The ID of the run to fetch metrics for
        
    Returns:
        Dict[str, List[Any]]: Dictionary mapping metric names to their history
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Get the history of all metrics
    history = run.history()
    
    # Convert to a more convenient format
    metrics = {}
    for column in history.columns:
        if column not in ['_timestamp', '_runtime', '_step']:
            metrics[column] = history[column].tolist()
    
    return metrics


@mcp.tool()
def query_wandb_run_system_metrics(entity: str, project: str, run_id: str) -> Dict[str, List[Any]]:
    """
    Fetch system metrics history for a specific run.
    
    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        run_id (str): The ID of the run to fetch metrics for
        
    Returns:
        Dict[str, List[Any]]: Dictionary mapping system metric names to their history
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Get the history of system metrics
    system_metrics = run.history(stream="events")
    
    # Convert to a more convenient format
    metrics = {}
    for column in system_metrics.columns:
        if column not in ['_timestamp', '_runtime', '_step']:
            metrics[column] = system_metrics[column].tolist()
    
    return metrics


@mcp.tool()
def query_wandb_run_summary_metrics(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch summary metrics for a specific run.
    
    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        run_id (str): The ID of the run to fetch metrics for
        
    Returns:
        Dict[str, Any]: Dictionary containing summary metrics
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return run.summary


@mcp.tool()
def query_wandb_artifacts(
    entity: str,
    project: str,
    artifact_name: Optional[str] = None,
    artifact_type: Optional[str] = None,
    version_alias: str = "latest"
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetches details for a specific artifact or lists artifact collections of a specific type.

    If artifact_name is provided, fetches details for that specific artifact.
    If artifact_name is not provided, artifact_type must be provided to list
    collections of that type.

    Args:
        entity (str): The wandb entity (username or team name).
        project (str): The project name.
        artifact_name (Optional[str]): The name of the artifact to fetch (e.g., 'my-dataset').
                                       If None, lists collections based on artifact_type.
        artifact_type (Optional[str]): The type of artifact collection to list.
                                       Required if artifact_name is None.
        version_alias (str): The version or alias for the specific artifact
                             (e.g., 'v1', 'latest'). Defaults to 'latest'.
                             Ignored if artifact_name is None.

    Returns:
        Union[List[Dict[str, Any]], Dict[str, Any]]:
            - Dict[str, Any]: Details of the specified artifact if artifact_name is provided.
            - List[Dict[str, Any]]: List of artifact collections if artifact_name is None
                                     and artifact_type is provided.

    Raises:
        ValueError: If neither artifact_name nor artifact_type is provided,
                    or if artifact_name is None and artifact_type is also None.
        wandb.errors.CommError: If the specified artifact is not found when artifact_name is provided.
    """
    api = wandb.Api()

    if artifact_name:
        # Fetch specific artifact details (logic from get_artifact)
        try:
            artifact = api.artifact(name=f"{entity}/{project}/{artifact_name}:{version_alias}")
            artifact_data = {
                "id": artifact.id,
                "name": artifact.name,
                "type": artifact.type,
                "version": artifact.version,
                "aliases": artifact.aliases,
                "state": artifact.state,
                "size": artifact.size,
                "created_at": artifact.created_at,
                "description": artifact.description,
                "metadata": artifact.metadata,
                "digest": artifact.digest,
            }
            return artifact_data
        except wandb.errors.CommError as e:
            # Re-raise to signal artifact not found or other communication issues
            raise e
    elif artifact_type:
        # List artifact collections (logic from list_artifact_collections)
        collections = api.artifact_collections(project_name=f"{entity}/{project}", type_name=artifact_type)
        collections_data = []
        for collection in collections:
            collections_data.append({
                "name": collection.name,
                "type": collection.type,
                "project": project, # Include project for clarity
                "entity": entity,   # Include entity for clarity
            })
        return collections_data
    else:
        raise ValueError("Either 'artifact_name' or 'artifact_type' must be provided.")



def query_wandb_sweeps(
    entity: str,
    project: str,
    action: str,
    sweep_id: Optional[str] = None
) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
    """
    Manages W&B sweeps: either lists all sweeps in a project OR gets the best run for a specific sweep.

    Use the 'action' parameter to specify the desired operation:
    - Set action='list_sweeps' to list all sweeps in the project. 'sweep_id' is ignored.
    - Set action='get_best_run' to find the best run for a specific sweep. 'sweep_id' is REQUIRED for this action.

    Args:
        entity (str): The wandb entity (username or team name).
        project (str): The project name.
        action (str): The operation to perform. Must be exactly 'list_sweeps' or 'get_best_run'.
        sweep_id (Optional[str]): The unique ID of the sweep. This is REQUIRED only when action='get_best_run'.
                                  It is ignored if action='list_sweeps'.

    Returns:
        Union[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
            - If action='list_sweeps': Returns a list of unique sweeps found in the project. [List[Dict]]
            - If action='get_best_run': Returns details of the best run for the specified sweep_id. [Dict]
                                        Returns None if the sweep exists but has no best run yet. [None]

    Raises:
        ValueError: If 'action' is not 'list_sweeps' or 'get_best_run'.
        ValueError: If action='get_best_run' but 'sweep_id' is not provided.
        wandb.errors.CommError: If a provided 'sweep_id' (when action='get_best_run') is not found or other API errors occur.
    """
    api = wandb.Api()

    if action == 'list_sweeps':
        # List all sweeps in the project (logic from original list_wandb_sweeps)
        runs = api.runs(f"{entity}/{project}", include_sweeps=True)
        sweeps_found = {}
        for run in runs:
            if run.sweep and run.sweep.id not in sweeps_found:
                sweep_obj = run.sweep
                sweeps_found[sweep_obj.id] = {
                    "id": sweep_obj.id,
                    "config": sweep_obj.config,
                    "metric": getattr(sweep_obj, 'metric', None),
                    "method": getattr(sweep_obj, 'method', None),
                    "entity": sweep_obj.entity,
                    "project": sweep_obj.project,
                    "state": sweep_obj.state,
                }
        return list(sweeps_found.values())

    elif action == 'get_best_run':
        # Get the best run for a specific sweep (logic from original get_wandb_sweep_best_run)
        if sweep_id is None:
            raise ValueError("The 'sweep_id' argument is required when action is 'get_best_run'.")

        try:
            sweep = api.sweep(path=f"{entity}/{project}/{sweep_id}")
            best_run = sweep.best_run()

            if best_run:
                run_dict = {
                    "id": best_run.id,
                    "name": best_run.name,
                    "state": best_run.state,
                    "config": best_run.config,
                    "summary": best_run.summary,
                    "created_at": best_run.created_at,
                    "url": best_run.url,
                    "tags": best_run.tags,
                }
                return run_dict
            else:
                # Sweep exists, but no best run found
                return None
        except wandb.errors.CommError as e:
            # Re-raise if sweep_id itself is invalid or other API error occurs
            raise e
    else:
        # Invalid action specified
        raise ValueError(f"Invalid action specified: '{action}'. Must be 'list_sweeps' or 'get_best_run'.")



def query_wandb_reports(entity: str, project: str) -> List[Dict[str, Any]]:
    """
    List available W&B Reports within a project.
    
    Args:
        entity (str): The wandb entity (username or team name)
        project (str): The project name
        
    Returns:
        List[Dict[str, Any]]: List of report dictionaries.
    """
    # Note: The public API for listing reports might be less direct.
    # `api.reports` might require entity/project to be set in Api() constructor
    # or might work differently. This is an attempt based on API structure.
    # If this fails, GraphQL might be necessary (see execute_graphql_query).
    try:
        # Initialize API potentially with overrides if needed
        api = wandb.Api(overrides={"entity": entity, "project": project})
        reports = api.reports() # Assumes this lists reports for the configured entity/project
        
        reports_data = []
        for report in reports:
            # Attributes depend on the actual Report object structure
            report_data = {
                "id": getattr(report, 'id', None), # Adjust attribute names as needed
                "name": getattr(report, 'name', None), 
                "title": getattr(report, 'title', getattr(report, 'display_name', None)),
                "description": getattr(report, 'description', None),
                "url": getattr(report, 'url', None), 
                "created_at": getattr(report, 'created_at', None),
                "updated_at": getattr(report, 'updated_at', None),
            }
            reports_data.append(report_data)
        return reports_data
    except Exception as e:
        # Consider logging the error
        print(f"Error listing reports for {entity}/{project}: {e}. Direct report listing might require GraphQL.")
        # Fallback or raise error
        return [] # Return empty list on error for now




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
