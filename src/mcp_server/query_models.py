import copy
import json
import logging
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Union

import wandb
from wandb_gql import gql

# Added imports for AST pagination
import copy
from graphql.language import ast as gql_ast, visitor as gql_visitor, printer as gql_printer, parse
from graphql.error import GraphQLError
from typing import Optional, List, Any # Ensure these are here

# Create a logger for this module
logger = logging.getLogger(__name__)



# def query_wandb_gql(query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
#     """
#     Execute an arbitrary GraphQL query against the Weights & Biases (W&B) API.
#     """
#     # Initialize wandb API, auth is handled via environment variables
#     api = wandb.Api()
#     result = api.client.execute(query, variable_values=variables or {})

#     return result


# # Helper function to find paginated gql collections in a response
# def find_paginated_collections(obj, path=""):
#     collections = []
#     if isinstance(obj, dict):
#         # Check if this object is a paginated collection
#         if "edges" in obj and "pageInfo" in obj and \
#             isinstance(obj.get("edges"), list) and \
#             isinstance(obj.get("pageInfo"), dict) and \
#             "hasNextPage" in obj.get("pageInfo", {}) and \
#             "endCursor" in obj.get("pageInfo", {}):
#             collections.append(path)
        
#         # Recursively check all child objects
#         for key, value in obj.items():
#             new_path = f"{path}.{key}" if path else key
#             collections.extend(find_paginated_collections(value, new_path))
            
#     return collections


def query_paginated_wandb_gql(
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    max_items: int = 100,
    items_per_page: int = 20,
    deduplicate: bool = True,
) -> Dict[str, Any]:
    """
    Execute a GraphQL query against the W&B API with pagination support using AST modification.
    Handles a single paginated field detected via the connection pattern.
    Modifies the result dictionary in-place.
    
    Args:
        query: The GraphQL query string. MUST include pageInfo{hasNextPage, endCursor} for paginated fields.
        variables: Variables to pass to the GraphQL query.
        max_items: Maximum number of items to fetch across all pages (default: 100).
        items_per_page: Number of items to request per page (default: 20).
        deduplicate: Whether to deduplicate nodes by ID across pages (default: True).
        
    Returns:
        The aggregated GraphQL response dictionary.
    """
    result_dict = {} 
    api = None 
    limit_key = None 
    try:
        api = wandb.Api() 
        logger.info("--- Inside query_paginated_wandb_gql: Step 0: Execute Initial Query ---")
        
        # Determine limit key and set initial page vars
        page1_vars_func = variables.copy() if variables is not None else {}
        limit_key = None 
        for k in page1_vars_func: 
            if k.lower() in ["limit", "first", "count"]: 
                limit_key = k
                break
        if limit_key:
             # Ensure first page uses items_per_page if limit is too high or missing
            page1_vars_func[limit_key] = min(items_per_page, page1_vars_func.get(limit_key) or items_per_page)
        else:
            limit_key = "limit" 
            page1_vars_func[limit_key] = items_per_page
            logger.debug(f"No limit variable found in input, adding '{limit_key}={items_per_page}'")

        # Parse for execution
        try:
             parsed_initial_query = gql(query.strip())
        except Exception as e:
             logger.error(f"Failed to parse initial query with wandb_gql: {e}")
             return {"errors": [{"message": f"Failed to parse initial query: {e}"}]}
        
        # Execute initial query
        try:
             result1 = api.client.execute(parsed_initial_query, variable_values=page1_vars_func)
             result_dict = copy.deepcopy(result1) # Work on a copy
             if "errors" in result_dict:
                  logger.error(f"GraphQL errors in initial response: {result_dict['errors']}")
                  return result_dict # Return errors if found
        except Exception as e:
            logger.error(f"Failed to execute initial GraphQL query: {e}", exc_info=True)
            return {"errors": [{"message": f"Failed to execute initial query: {e}"}]}

        # Find Collections
        detected_paths = find_paginated_collections(result_dict)
        if not detected_paths:
            logger.info("No paginated paths detected. Returning initial result.")
            return result_dict 
        
        # --- Use the first detected path --- 
        # TODO: Enhance to handle multiple paths if necessary
        path_to_paginate = detected_paths[0] 
        logger.info(f"Using path for pagination: {'/'.join(path_to_paginate)}")
        
        # Extract page 1 data
        runs_data1 = get_nested_value(result_dict, path_to_paginate)
        if runs_data1 is None: 
             logger.warning(f"Could not extract data for pagination path {'/'.join(path_to_paginate)}. Returning initial result.")
             return result_dict 
        page_info1 = get_nested_value(runs_data1, ['pageInfo'])
        if page_info1 is None: 
             logger.warning(f"Could not extract pageInfo for pagination path {'/'.join(path_to_paginate)}. Returning initial result.")
             return result_dict 

        cursor = page_info1.get("endCursor")
        has_next = page_info1.get("hasNextPage")
        initial_edges = runs_data1.get('edges', []) 
        logging.info(f"Page 1 Results: {len(initial_edges)} runs.")
        logging.info(f"Page 1 PageInfo: {page_info1}")

        # Deduplicate initial edges and update result_dict
        seen_ids = set() 
        current_edge_count = 0
        temp_initial_edges = [] 
        if initial_edges:
            for edge in initial_edges:
                 try:
                     # Check max items even on page 1 relative to the limit
                     if current_edge_count >= max_items: break
                     node_id = edge['node']['id']
                     if node_id not in seen_ids:
                         seen_ids.add(node_id)
                         temp_initial_edges.append(edge) 
                         current_edge_count += 1
                 except (KeyError, TypeError):
                     if current_edge_count < max_items:
                          temp_initial_edges.append(edge) 
                          current_edge_count += 1
            # Update the edges in the result_dict 
            target_collection_dict = get_nested_value(result_dict, path_to_paginate)
            if target_collection_dict:
                target_collection_dict['edges'] = temp_initial_edges[:max_items] # Ensure initial list respects max_items
                current_edge_count = len(target_collection_dict['edges'])
            logging.info(f"Stored {current_edge_count} unique edges after page 1 (max: {max_items}).")
        
        if not has_next or not cursor or current_edge_count >= max_items:
            logger.info("No further pages needed based on page 1 info or max_items reached.")
            # Ensure final pageInfo reflects reality
            target_pi_dict = get_nested_value(result_dict, path_to_paginate + ['pageInfo'])
            if target_pi_dict:
                 target_pi_dict["hasNextPage"] = False
            return result_dict 

        # Generate Paginated Query String
        logging.info("\n--- Generating Paginated Query String --- ")
        generated_paginated_query_string = None 
        after_variable_name = "after" # Standard name
        try:
            initial_ast = parse(query.strip())
            visitor = AddPaginationArgsVisitor(
                field_paths=detected_paths, 
                first_variable_name=limit_key, 
                after_variable_name=after_variable_name
            ) 
            modified_ast = gql_visitor.visit(copy.deepcopy(initial_ast), visitor)
            generated_paginated_query_string = gql_printer.print_ast(modified_ast)
            logger.info("AST modification and printing successful.")
        except Exception as e:
             logger.error(f"Failed to generate query string via AST: {e}", exc_info=True)
             return result_dict # Return what we have if generation fails
             
        if generated_paginated_query_string is None: return result_dict

        logging.info("\n--- Loop: Execute, Deduplicate, Aggregate In-Place, Check Limit ---")
        page_num = 1 
        current_cursor = cursor 
        current_has_next = has_next
        final_page_info = page_info1 
        
        while current_has_next:
            if current_edge_count >= max_items:
                logging.info(f"Reached max_items ({max_items}). Stopping loop.")
                final_page_info = {**final_page_info, "hasNextPage": False}
                break
            
            page_num += 1
            logging.info(f"\nFetching Page {page_num}...")
            page_vars = variables.copy() if variables is not None else {} # Start with original vars
            page_vars[limit_key] = items_per_page # Set correct page size
            page_vars[after_variable_name] = current_cursor # Set cursor
            
            try:
                # Parse and execute for the current page
                parsed_generated = gql(generated_paginated_query_string) 
                logging.info(f"Executing generated query for page {page_num} with vars: {page_vars}")
                result_page = api.client.execute(parsed_generated, variable_values=page_vars)
                
                if "errors" in result_page:
                    logger.error(f"GraphQL errors on page {page_num}: {result_page['errors']}. Stopping pagination.")
                    current_has_next = False
                    final_page_info = {**final_page_info, "hasNextPage": False} # Update page info on error
                    continue # Go to end of loop
                
                runs_data = get_nested_value(result_page, path_to_paginate)
                if runs_data is None:
                    logging.warning(f"Could not get data for path {'/'.join(path_to_paginate)} on page {page_num}. Stopping.")
                    current_has_next = False 
                    continue 
                else:
                    edges_this_page = get_nested_value(runs_data, ['edges']) or []
                    page_info = get_nested_value(runs_data, ['pageInfo']) or {}
                    final_page_info = page_info # Store latest page info
                
                logging.info(f"Result (Page {page_num}): {len(edges_this_page)} runs returned.")
                logging.info(f"Page Info (Page {page_num}): {page_info}")

                # Deduplicate & Find edges to append
                new_edges_for_aggregation = []
                duplicates_skipped = 0
                if edges_this_page:
                    for edge in edges_this_page:
                        if current_edge_count + len(new_edges_for_aggregation) >= max_items:
                            logging.info(f"Max items ({max_items}) reached mid-page {page_num}.")
                            final_page_info = {**final_page_info, "hasNextPage": False}
                            current_has_next = False 
                            break 
                        
                        try:
                            node_id = edge['node']['id']
                            if node_id not in seen_ids:
                                seen_ids.add(node_id)
                                new_edges_for_aggregation.append(edge)
                            else:
                                duplicates_skipped += 1
                        except (KeyError, TypeError):
                             new_edges_for_aggregation.append(edge) 
                    
                    if duplicates_skipped > 0:
                        logging.info(f"Skipped {duplicates_skipped} duplicate edges on page {page_num}.")
                        
                    # Append new unique edges IN-PLACE
                    if new_edges_for_aggregation:
                        target_collection_dict_inplace = get_nested_value(result_dict, path_to_paginate)
                        if target_collection_dict_inplace and isinstance(target_collection_dict_inplace.get('edges'), list):
                            target_collection_dict_inplace['edges'].extend(new_edges_for_aggregation)
                            current_edge_count = len(target_collection_dict_inplace['edges'])
                            logging.info(f"Appended {len(new_edges_for_aggregation)} new edges. Total unique edges: {current_edge_count}")
                        else:
                            logging.error("Could not find target edges list in result_dict to append in-place.")
                            current_has_next = False 
                    else:
                        if len(edges_this_page) > 0:
                            logging.info("No new unique edges found on page {page_num} after deduplication.")
                        else:
                            logging.info("No edges returned on page {page_num} to aggregate.")
                else:
                    logging.info("No edges returned on page {page_num} to aggregate.")
                
                # Update cursor and has_next for next loop iteration (or final state)
                current_cursor = final_page_info.get("endCursor")
                # Respect hasNextPage from API unless loop was broken early by max_items or errors
                if current_has_next: # Only update if loop didn't break mid-page 
                     current_has_next = final_page_info.get("hasNextPage", False)

                # Safety checks
                if current_has_next and not current_cursor:
                    logging.warning(f"hasNextPage is true but no endCursor received. Stopping loop.")
                    current_has_next = False
                if not edges_this_page:
                     logging.warning(f"No edges received for page {page_num}. Stopping loop.")
                     current_has_next = False 

            except Exception as e:
                logging.error(f"Execution failed for page {page_num}: {e}", exc_info=True)
                current_has_next = False # Stop loop on error

        logging.info(f"\n--- Pagination Loop Finished after page {page_num} ---")
        logging.info(f"Final aggregated edge count: {current_edge_count}") 
        
        # Update the final pageInfo in the result dictionary
        target_collection_dict_final = get_nested_value(result_dict, path_to_paginate)
        if target_collection_dict_final:
             target_collection_dict_final['pageInfo'] = final_page_info
             logging.info(f"Updated final pageInfo: {final_page_info}")

        return result_dict # Return the modified dictionary

    except Exception as e:
        error_message = f"Critical error in paginated GraphQL query function: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        # Return original dict if possible, else error structure
        if result_dict: 
             if "errors" not in result_dict: result_dict["errors"] = []
             result_dict["errors"].append({"message": "Pagination failed", "details": str(e)})
             return result_dict
        else:
             return {"errors": [{"message": "Pagination failed catastrophically", "details": str(e)}]}


def find_paginated_collections(obj: Dict, current_path: Optional[List[str]] = None) -> List[List[str]]:
    """Find collections in a response that follow the W&B connection pattern. Returns List[List[str]]."""
    # Ensure this implementation correctly builds and returns List[List[str]]
    if current_path is None: current_path = []
    collections = []
    if isinstance(obj, dict):
        if (
            "edges" in obj and "pageInfo" in obj and
            isinstance(obj.get("edges"), list) and
            isinstance(obj.get("pageInfo"), dict) and
            "hasNextPage" in obj.get("pageInfo", {}) and
            "endCursor" in obj.get("pageInfo", {})
        ):
            collections.append(list(current_path)) # Correct: append list path
        # Recurse correctly
        for key, value in obj.items():
            current_path.append(key)
            collections.extend(find_paginated_collections(value, current_path))
            current_path.pop()
    elif isinstance(obj, list):
         for item in obj:
             collections.extend(find_paginated_collections(item, current_path))
    return collections


def get_nested_value(obj: Dict, path: list[str]) -> Optional[Any]:
    """Get a value from a nested dictionary using a list of keys (path)."""
    current = obj
    # Iterate directly over the list path
    for key in path: 
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def list_entity_projects(entity: str = None) -> List[Dict[str, Any]]:
    """
    Fetch all projects for a specific wandb entity. If no entity is provided, 
    fetches projects for the current user and their teams.

    Args:
        entity (str, optional): The wandb entity (username or team name). If None, 
                               fetches projects for the current user and their teams.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping entity names to lists of project dictionaries.
            Each project dictionary contains:
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

    # Merge entity and teams into a single list
    if entity is None:
        viewer = api.viewer
        entities = [viewer.entity] + viewer.teams
    else:
        entities = [entity]

    # Get all projects for the entity

    entities_projects = {}
    for entity in entities:
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

        entities_projects[entity] = projects_data

    return entities_projects


def query_wandb_runs(
    entity: str,
    project: str,
    per_page: int = 50,
    order: str = "-created_at",
    filters: Dict[str, Any] = None,
    search: str = None,
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
    query_params = {"per_page": per_page, "order": order}

    # Add filters if provided
    if filters:
        for key, value in filters.items():
            if key in ["state", "tags", "config", "summary"]:
                query_params[key] = value

    # Add search if provided
    if search:
        query_params["search"] = search

    # Get runs from the specified entity and project with filters
    runs = api.runs(f"{entity}/{project}", **query_params)

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


def query_wandb_run_training_metrics(
    entity: str, project: str, run_id: str
) -> Dict[str, List[Any]]:
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
        if column not in ["_timestamp", "_runtime", "_step"]:
            metrics[column] = history[column].tolist()

    return metrics


def query_wandb_run_system_metrics(
    entity: str, project: str, run_id: str
) -> Dict[str, List[Any]]:
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
        if column not in ["_timestamp", "_runtime", "_step"]:
            metrics[column] = system_metrics[column].tolist()

    return metrics


def query_wandb_run_summary_metrics(
    entity: str, project: str, run_id: str
) -> Dict[str, Any]:
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


def query_wandb_artifacts(
    entity: str,
    project: str,
    artifact_name: Optional[str] = None,
    artifact_type: Optional[str] = None,
    version_alias: str = "latest",
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
            artifact = api.artifact(
                name=f"{entity}/{project}/{artifact_name}:{version_alias}"
            )
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
        collections = api.artifact_collections(
            project_name=f"{entity}/{project}", type_name=artifact_type
        )
        collections_data = []
        for collection in collections:
            collections_data.append(
                {
                    "name": collection.name,
                    "type": collection.type,
                    "project": project,  # Include project for clarity
                    "entity": entity,  # Include entity for clarity
                }
            )
        return collections_data
    else:
        raise ValueError("Either 'artifact_name' or 'artifact_type' must be provided.")


def query_wandb_sweeps(
    entity: str, project: str, action: str, sweep_id: Optional[str] = None
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

    if action == "list_sweeps":
        # List all sweeps in the project (logic from original list_wandb_sweeps)
        runs = api.runs(f"{entity}/{project}", include_sweeps=True)
        sweeps_found = {}
        for run in runs:
            if run.sweep and run.sweep.id not in sweeps_found:
                sweep_obj = run.sweep
                sweeps_found[sweep_obj.id] = {
                    "id": sweep_obj.id,
                    "config": sweep_obj.config,
                    "metric": getattr(sweep_obj, "metric", None),
                    "method": getattr(sweep_obj, "method", None),
                    "entity": sweep_obj.entity,
                    "project": sweep_obj.project,
                    "state": sweep_obj.state,
                }
        return list(sweeps_found.values())

    elif action == "get_best_run":
        # Get the best run for a specific sweep (logic from original get_wandb_sweep_best_run)
        if sweep_id is None:
            raise ValueError(
                "The 'sweep_id' argument is required when action is 'get_best_run'."
            )

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
        raise ValueError(
            f"Invalid action specified: '{action}'. Must be 'list_sweeps' or 'get_best_run'."
        )


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
        reports = (
            api.reports()
        )  # Assumes this lists reports for the configured entity/project

        reports_data = []
        for report in reports:
            # Attributes depend on the actual Report object structure
            report_data = {
                "id": getattr(report, "id", None),  # Adjust attribute names as needed
                "name": getattr(report, "name", None),
                "title": getattr(
                    report, "title", getattr(report, "display_name", None)
                ),
                "description": getattr(report, "description", None),
                "url": getattr(report, "url", None),
                "created_at": getattr(report, "created_at", None),
                "updated_at": getattr(report, "updated_at", None),
            }
            reports_data.append(report_data)
        return reports_data
    except Exception as e:
        # Consider logging the error
        print(
            f"Error listing reports for {entity}/{project}: {e}. Direct report listing might require GraphQL."
        )
        # Fallback or raise error
        return []  # Return empty list on error for now

# --- AST Visitor --- 
class AddPaginationArgsVisitor(gql_visitor.Visitor):
    """ Adds first/after args and variables """
    def __init__(self, field_paths, first_variable_name="limit", after_variable_name="after"):
        super().__init__()
        self.field_paths = set(tuple(p) for p in field_paths)
        self.first_variable_name = first_variable_name
        self.after_variable_name = after_variable_name
        self.current_path = []
        self.modified_operation = False

    def enter_field(self, node, key, parent, path, ancestors):
        field_name = node.alias.value if node.alias else node.name.value
        self.current_path.append(field_name)
        current_path_tuple = tuple(self.current_path)
        if current_path_tuple in self.field_paths:
            existing_args = list(node.arguments)
            args_changed = False
            has_first = any(arg.name.value == "first" for arg in existing_args)
            if not has_first:
                # Defaulting variable name to 'limit' if not found, might need refinement
                limit_var_node = gql_ast.VariableNode(name=gql_ast.NameNode(value=self.first_variable_name))
                existing_args.append(gql_ast.ArgumentNode(name=gql_ast.NameNode(value="first"), value=limit_var_node))
                args_changed = True
            has_after = any(arg.name.value == "after" for arg in existing_args)
            if not has_after:
                 existing_args.append(gql_ast.ArgumentNode(name=gql_ast.NameNode(value="after"), value=gql_ast.VariableNode(name=gql_ast.NameNode(value=self.after_variable_name))))
                 args_changed = True
            if args_changed:
                node.arguments = tuple(existing_args)

    def leave_field(self, node, key, parent, path, ancestors):
        if self.current_path:
            self.current_path.pop()

    def enter_operation_definition(self, node, key, parent, path, ancestors):
        if self.modified_operation: return
        existing_vars = {var.variable.name.value for var in node.variable_definitions}
        new_defs_list = list(node.variable_definitions)
        defs_changed = False
        # Determine limit variable name from existing vars if possible, else default
        current_limit_var = self.first_variable_name # Default
        for var_name in existing_vars:
             if var_name.lower() in ["limit", "first", "count"]:
                  current_limit_var = var_name
                  break
        
        if current_limit_var not in existing_vars:
             new_defs_list.append(gql_ast.VariableDefinitionNode(variable=gql_ast.VariableNode(name=gql_ast.NameNode(value=current_limit_var)), type=gql_ast.NamedTypeNode(name=gql_ast.NameNode(value="Int"))))
             defs_changed = True
        if self.after_variable_name not in existing_vars:
             new_defs_list.append(gql_ast.VariableDefinitionNode(variable=gql_ast.VariableNode(name=gql_ast.NameNode(value=self.after_variable_name)), type=gql_ast.NamedTypeNode(name=gql_ast.NameNode(value="String"))))
             defs_changed = True
        if defs_changed:
            node.variable_definitions = tuple(new_defs_list)
        self.modified_operation = True

# --- End AST Visitor & Helpers ---
