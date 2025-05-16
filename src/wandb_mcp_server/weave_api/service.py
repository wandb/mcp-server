"""
Service layer for Weave API.

This module provides high-level services for querying and processing Weave traces.
It orchestrates the client, query builder, and processor components.
"""

from typing import Any, Dict, List, Optional, Set

from wandb_mcp_server.weave_api.client import WeaveApiClient
from wandb_mcp_server.weave_api.models import QueryResult
from wandb_mcp_server.weave_api.processors import TraceProcessor
from wandb_mcp_server.weave_api.query_builder import QueryBuilder
from wandb_mcp_server.utils import get_rich_logger

# Import CallSchema to validate column names
try:
    from weave.trace_server.trace_server_interface import CallSchema
    VALID_COLUMNS = set(CallSchema.__annotations__.keys())
    HAVE_CALL_SCHEMA = True
except ImportError:
    # Fallback if CallSchema isn't available
    VALID_COLUMNS = {
        "id", "project_id", "op_name", "display_name", "trace_id", "parent_id", 
        "started_at", "attributes", "inputs", "ended_at", "exception", "output", 
        "summary", "wb_user_id", "wb_run_id", "deleted_at", "storage_size_bytes",
        "total_storage_size_bytes"
    }
    HAVE_CALL_SCHEMA = False

logger = get_rich_logger(__name__)


class TraceService:
    """Service for querying and processing Weave traces."""

    # Define cost fields once as a class constant
    COST_FIELDS = {"total_cost", "completion_cost", "prompt_cost"}
    
    # Define synthetic columns that shouldn't be passed to the API but can be reconstructed
    SYNTHETIC_COLUMNS = {"costs"}
    
    # Define latency field mapping
    LATENCY_FIELD_MAPPING = {
        "latency_ms": "summary.weave.latency_ms"
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        server_url: Optional[str] = None,
        retries: int = 3,
        timeout: int = 10,
    ):
        """Initialize the Trace service.

        Args:
            api_key: Weights & Biases API key. If not provided, will try to get from environment.
            server_url: Weave API server URL. If not provided, will try to get from environment.
            retries: Number of retries for HTTP requests.
            timeout: Timeout for HTTP requests in seconds.
        """
        self.client = WeaveApiClient(
            api_key=api_key,
            server_url=server_url,
            retries=retries,
            timeout=timeout,
        )
        
        # Initialize collection for invalid columns (for warning messages)
        self.invalid_columns = set()

    def _map_latency_columns(self, columns: Optional[List[str]]) -> Optional[List[str]]:
        """Map latency columns to their proper path.
        
        Args:
            columns: List of columns.
            
        Returns:
            Mapped columns.
        """
        if not columns:
            return columns
            
        mapped_columns = []
        for col in columns:
            if col in self.LATENCY_FIELD_MAPPING:
                mapped_col = self.LATENCY_FIELD_MAPPING[col]
                logger.info(f"Mapping column '{col}' to '{mapped_col}'")
                mapped_columns.append(mapped_col)
            else:
                mapped_columns.append(col)
                
        return mapped_columns
        
    def _validate_and_filter_columns(self, columns: Optional[List[str]]) -> tuple[Optional[List[str]], List[str], Set[str]]:
        """Validate columns against CallSchema and filter out synthetic/invalid columns.
        
        Args:
            columns: List of columns.
            
        Returns:
            Tuple of (filtered_columns, requested_synthetic_columns, invalid_columns)
        """
        if not columns:
            return columns, [], set()
            
        filtered_columns = []
        requested_synthetic_columns = []
        invalid_columns = set()
        
        # Define synthetic fields that should be kept in the columns list
        synthetic_fields_to_keep = {"status", "latency_ms"}
        
        for col in columns:
            if col in self.SYNTHETIC_COLUMNS:
                logger.info(f"Filtering out synthetic column '{col}' from API request")
                requested_synthetic_columns.append(col)
            elif col in synthetic_fields_to_keep:
                # Keep these in filtered columns AND treat as synthetic
                filtered_columns.append(col)
                requested_synthetic_columns.append(col)
            elif col in self.LATENCY_FIELD_MAPPING:
                # Simply map to the source field
                # Don't add to filtered_columns, but keep track for synthetic generation
                requested_synthetic_columns.append(col)
                # We also need the summary field for this, so make sure it's included
                if "summary" not in filtered_columns:
                    filtered_columns.append("summary")
            elif col in VALID_COLUMNS:
                # Keep valid columns
                filtered_columns.append(col)
            else:
                logger.warning(f"Invalid column '{col}' requested and will be ignored")
                invalid_columns.add(col)
                
        return filtered_columns, requested_synthetic_columns, invalid_columns
    
    def _ensure_required_columns_for_synthetic(self, filtered_columns: Optional[List[str]], requested_synthetic_columns: List[str]) -> Optional[List[str]]:
        """Ensure required columns for synthetic fields are included.
        
        Args:
            filtered_columns: List of columns after filtering out synthetic ones.
            requested_synthetic_columns: List of requested synthetic columns.
            
        Returns:
            Updated filtered columns list with required columns added.
        """
        if not filtered_columns:
            filtered_columns = []
            
        required_columns = set(filtered_columns)
        
        # Add required columns for synthesizing costs
        if "costs" in requested_synthetic_columns:
            # Costs data comes from summary.weave.costs
            if "summary" not in required_columns:
                logger.info("Adding 'summary' column as it's required for costs data")
                required_columns.add("summary")
        
        # Add other required columns for other synthetic fields as needed
        
        return list(required_columns)
    
    def _add_synthetic_columns(self, traces: List[Dict[str, Any]], requested_synthetic_columns: List[str], invalid_columns: Set[str]) -> List[Dict[str, Any]]:
        """Add synthetic columns back to the traces and add warnings for invalid columns.
        
        Args:
            traces: List of trace dictionaries.
            requested_synthetic_columns: List of requested synthetic columns.
            invalid_columns: Set of invalid column names that were requested.
            
        Returns:
            Updated traces with synthetic columns added and invalid column warnings.
        """
        if not requested_synthetic_columns and not invalid_columns:
            return traces
            
        updated_traces = []
        
        for trace in traces:
            updated_trace = trace.copy()
            
            # Add costs data if requested
            if "costs" in requested_synthetic_columns:
                costs_data = trace.get("summary", {}).get("weave", {}).get("costs", {})
                if costs_data:
                    logger.debug(f"Adding synthetic 'costs' column with {len(costs_data)} providers")
                    updated_trace["costs"] = costs_data
                else:
                    logger.warning(f"No costs data found in trace {trace.get('id')}")
                    updated_trace["costs"] = {}
            
            # Add status from summary if requested
            if "status" in requested_synthetic_columns:
                status = trace.get("status")  # Check if it's already in the trace
                if not status:
                    # Extract from summary.weave.status
                    status = trace.get("summary", {}).get("weave", {}).get("status")
                    if status:
                        logger.debug(f"Adding synthetic 'status' from summary: {status}")
                        updated_trace["status"] = status
                    else:
                        logger.warning(f"No status data found in trace {trace.get('id')}")
                        updated_trace["status"] = None
            
            # Add latency_ms from summary if requested
            if "latency_ms" in requested_synthetic_columns:
                latency = trace.get("latency_ms")  # Check if it's already in the trace
                if latency is None:
                    # Extract from summary.weave.latency_ms
                    latency = trace.get("summary", {}).get("weave", {}).get("latency_ms")
                    if latency is not None:
                        logger.debug(f"Adding synthetic 'latency_ms' from summary: {latency}")
                        updated_trace["latency_ms"] = latency
                    else:
                        logger.warning(f"No latency_ms data found in trace {trace.get('id')}")
                        updated_trace["latency_ms"] = None
            
            # Add warnings for invalid columns
            for col in invalid_columns:
                warning_message = f"{col} is not a valid column name, no data returned"
                updated_trace[col] = warning_message
            
            updated_traces.append(updated_trace)
            
        return updated_traces

    def query_traces(
        self,
        entity_name: str,
        project_name: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "started_at",
        sort_direction: str = "desc",
        limit: Optional[int] = None,
        offset: int = 0,
        include_costs: bool = True,
        include_feedback: bool = True,
        columns: Optional[List[str]] = None,
        expand_columns: Optional[List[str]] = None,
        truncate_length: Optional[int] = 200,
        return_full_data: bool = False,
        metadata_only: bool = False,
    ) -> QueryResult:
        """Query traces from the Weave API.

        Args:
            entity_name: Weights & Biases entity name.
            project_name: Weights & Biands project name.
            filters: Dictionary of filter conditions.
            sort_by: Field to sort by.
            sort_direction: Sort direction ('asc' or 'desc').
            limit: Maximum number of results to return.
            offset: Number of results to skip (for pagination).
            include_costs: Include tracked API cost information in the results.
            include_feedback: Include Weave annotations in the results.
            columns: List of specific columns to include in the results.
            expand_columns: List of columns to expand in the results.
            truncate_length: Maximum length for string values.
            return_full_data: Whether to include full untruncated trace data.
            metadata_only: Whether to only include metadata without traces.

        Returns:
            QueryResult object with metadata and optionally traces.
        """
        # Clear invalid columns from previous requests
        self.invalid_columns = set()
        
        # Special handling for cost-based sorting
        client_side_cost_sort = sort_by in self.COST_FIELDS
        
        # Handle latency field mapping
        if sort_by in self.LATENCY_FIELD_MAPPING:
            logger.info(f"Mapping sort field '{sort_by}' to '{self.LATENCY_FIELD_MAPPING[sort_by]}'")
            server_sort_by = self.LATENCY_FIELD_MAPPING[sort_by]
            server_sort_direction = sort_direction
        # If we need to do client-side sorting by cost, we need to ensure the costs are included
        elif client_side_cost_sort:
            include_costs = True
            # We need to override server-side sorting to something reasonable
            server_sort_by = "started_at"
            server_sort_direction = sort_direction
        elif sort_by not in VALID_COLUMNS:
            logger.warning(f"Invalid sort field '{sort_by}', falling back to 'started_at'")
            server_sort_by = "started_at"
            server_sort_direction = sort_direction
        else:
            server_sort_by = sort_by
            server_sort_direction = sort_direction
            
        # Map latency columns to their proper path
        mapped_columns = self._map_latency_columns(columns)
        
        # Validate and filter columns using CallSchema
        filtered_columns, requested_synthetic_columns, invalid_columns = self._validate_and_filter_columns(mapped_columns)
        
        # Store invalid columns for later
        self.invalid_columns = invalid_columns
        
        # If costs was requested as a column, make sure to include it
        if "costs" in requested_synthetic_columns:
            include_costs = True
        
        # Manually add latency_ms to synthetic fields if requested
        if columns and "latency_ms" in columns and "latency_ms" not in requested_synthetic_columns:
            requested_synthetic_columns.append("latency_ms")
            
        # Ensure required columns for synthetic fields are included
        filtered_columns = self._ensure_required_columns_for_synthetic(filtered_columns, requested_synthetic_columns)
        
        # Prepare query parameters
        query_params = {
            "entity_name": entity_name,
            "project_name": project_name,
            "filters": filters or {},
            "sort_by": server_sort_by,
            "sort_direction": server_sort_direction,
            "limit": None if client_side_cost_sort else limit,  # No limit if we're sorting by cost
            "offset": offset,
            "include_costs": include_costs,
            "include_feedback": include_feedback,
            "columns": filtered_columns,
            "expand_columns": expand_columns,
        }
        
        # Build request body
        request_body = QueryBuilder.prepare_query_params(query_params)
        
        # Extract synthetic fields if any were specified
        synthetic_fields = request_body.pop("_synthetic_fields", []) if "_synthetic_fields" in request_body else []
        
        # Make sure all requested synthetic columns are included in synthetic_fields
        for col in requested_synthetic_columns:
            if col not in synthetic_fields:
                synthetic_fields.append(col)
        
        # Execute query
        all_traces = list(self.client.query_traces(request_body))
        
        # Add synthetic columns and invalid column warnings back to the results
        if requested_synthetic_columns or invalid_columns:
            all_traces = self._add_synthetic_columns(all_traces, requested_synthetic_columns, invalid_columns)
        
        # Client-side cost-based sorting if needed
        if client_side_cost_sort and all_traces:
            logger.info(f"Performing client-side sorting by {sort_by}")
            # Sort traces by cost
            all_traces.sort(
                key=lambda t: TraceProcessor.get_cost(t, sort_by),
                reverse=(sort_direction == "desc")
            )
            # Apply limit if specified
            if limit is not None:
                all_traces = all_traces[:limit]
        
        # If we need to synthesize fields, do it
        if synthetic_fields:
            logger.info(f"Synthesizing fields: {synthetic_fields}")
            all_traces = [
                TraceProcessor.synthesize_fields(trace, synthetic_fields)
                for trace in all_traces
            ]
        
        # Process traces
        result = TraceProcessor.process_traces(
            traces=all_traces,
            truncate_length=truncate_length or 0,
            return_full_data=return_full_data,
            metadata_only=metadata_only,
        )
        
        return result
        
    def query_paginated_traces(
        self,
        entity_name: str,
        project_name: str,
        chunk_size: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "started_at",
        sort_direction: str = "desc",
        target_limit: Optional[int] = None,
        include_costs: bool = True,
        include_feedback: bool = True,
        columns: Optional[List[str]] = None,
        expand_columns: Optional[List[str]] = None,
        truncate_length: Optional[int] = 200,
        return_full_data: bool = False,
        metadata_only: bool = False,
    ) -> QueryResult:
        """Query traces with pagination.

        Args:
            entity_name: Weights & Biases entity name.
            project_name: Weights & Biands project name.
            chunk_size: Number of traces to retrieve in each chunk.
            filters: Dictionary of filter conditions.
            sort_by: Field to sort by.
            sort_direction: Sort direction ('asc' or 'desc').
            target_limit: Maximum total number of results to return.
            include_costs: Include tracked API cost information in the results.
            include_feedback: Include Weave annotations in the results.
            columns: List of specific columns to include in the results.
            expand_columns: List of columns to expand in the results.
            truncate_length: Maximum length for string values.
            return_full_data: Whether to include full untruncated trace data.
            metadata_only: Whether to only include metadata without traces.

        Returns:
            QueryResult object with metadata and optionally traces.
        """
        # Special handling for cost-based sorting
        client_side_cost_sort = sort_by in self.COST_FIELDS
        
        # Map latency_ms to summary.weave.latency_ms if needed
        effective_sort_by = self.LATENCY_FIELD_MAPPING.get(sort_by, sort_by)
        if sort_by in self.LATENCY_FIELD_MAPPING:
            logger.info(f"Mapping sort field '{sort_by}' to '{effective_sort_by}' for paginated query")
        elif sort_by not in VALID_COLUMNS and sort_by not in self.COST_FIELDS:
            logger.warning(f"Invalid sort field '{sort_by}', falling back to 'started_at'")
            effective_sort_by = "started_at"
            
        # Map latency columns
        mapped_columns = self._map_latency_columns(columns)
        
        # Validate and filter columns using CallSchema
        filtered_columns, requested_synthetic_columns, invalid_columns = self._validate_and_filter_columns(mapped_columns)
        
        # Store invalid columns for later
        self.invalid_columns = invalid_columns
        
        # If costs was requested as a column, make sure to include it
        if "costs" in requested_synthetic_columns:
            include_costs = True
            
        # Ensure required columns for synthetic fields are included
        filtered_columns = self._ensure_required_columns_for_synthetic(filtered_columns, requested_synthetic_columns)

        if client_side_cost_sort:
            logger.info(f"Cost-based sorting detected: {sort_by}")
            all_traces = self._query_for_cost_sorting(
                entity_name=entity_name,
                project_name=project_name,
                filters=filters,
                sort_by=sort_by,
                sort_direction=sort_direction,
                target_limit=target_limit,
                columns=filtered_columns,  # Pass filtered columns
                expand_columns=expand_columns,
                include_costs=True,  # Force include_costs for cost sorting
                include_feedback=include_feedback,
                requested_synthetic_columns=requested_synthetic_columns,  # Pass synthetic columns request
                invalid_columns=invalid_columns,  # Pass invalid columns
            )
        else:
            # Normal paginated query logic
            all_traces = []
            current_offset = 0
            
            while True:
                logger.info(f"Querying chunk with offset {current_offset}, size {chunk_size}")
                remaining = target_limit - len(all_traces) if target_limit else chunk_size
                current_chunk_size = min(chunk_size, remaining) if target_limit else chunk_size
                
                chunk_result = self.query_traces(
                    entity_name=entity_name,
                    project_name=project_name,
                    filters=filters,
                    sort_by=effective_sort_by,  # Use the mapped field name if applicable
                    sort_direction=sort_direction,
                    limit=current_chunk_size,
                    offset=current_offset,
                    include_costs=include_costs,
                    include_feedback=include_feedback,
                    columns=mapped_columns,  # Pass original (unmapped) columns to ensure proper handling
                    expand_columns=expand_columns,
                    return_full_data=True,  # We want raw data for now
                    metadata_only=False,
                )
                
                # Get the traces from the QueryResult and handle both None and empty list cases
                traces_from_chunk = chunk_result.traces if chunk_result and chunk_result.traces else []
                if not traces_from_chunk:
                    break
                    
                all_traces.extend(traces_from_chunk)
                
                if len(traces_from_chunk) < current_chunk_size or (
                    target_limit and len(all_traces) >= target_limit
                ):
                    break
                    
                current_offset += chunk_size
        
        # Process all traces at once with appropriate parameters
        if target_limit and all_traces:
            all_traces = all_traces[:target_limit]
        
        result = TraceProcessor.process_traces(
            traces=all_traces,
            truncate_length=truncate_length or 0,
            return_full_data=return_full_data,
            metadata_only=metadata_only,
        )
        logger.debug(f"Final result from query_paginated_traces:\n\n{len(result.model_dump_json(indent=2))}\n")
        return result
        
    def _query_for_cost_sorting(
        self,
        entity_name: str,
        project_name: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "total_cost",
        sort_direction: str = "desc",
        target_limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
        expand_columns: Optional[List[str]] = None,
        include_costs: bool = True,
        include_feedback: bool = True,
        requested_synthetic_columns: Optional[List[str]] = None,
        invalid_columns: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Special two-stage query logic for cost-based sorting.

        Args:
            entity_name: Weights & Biases entity name.
            project_name: Weights & Biands project name.
            filters: Dictionary of filter conditions.
            sort_by: Cost field to sort by.
            sort_direction: Sort direction ('asc' or 'desc').
            target_limit: Maximum number of results to return.
            columns: List of specific columns to include in the results.
            expand_columns: List of columns to expand in the results.
            include_costs: Include tracked API cost information in the results.
            include_feedback: Include Weave annotations in the results.
            requested_synthetic_columns: List of synthetic columns requested by the user.
            invalid_columns: Set of invalid column names that were requested.

        Returns:
            List of trace dictionaries sorted by the specified cost field.
        """
        if invalid_columns is None:
            invalid_columns = set()
            
        # First pass: Fetch all trace IDs and costs
        first_pass_query = {
            "entity_name": entity_name,
            "project_name": project_name,
            "filters": filters or {},
            "sort_by": "started_at",  # Use a standard sort for the first pass
            "sort_direction": "desc",
            "limit": 1000000,  # Explicitly set a large limit to get all traces
            "include_costs": True,  # We need costs for sorting
            "include_feedback": False,  # Don't need feedback for the first pass
            "columns": ["id", "summary"],  # Need summary for costs data
        }
        
        first_pass_request = QueryBuilder.prepare_query_params(first_pass_query)
        first_pass_results = list(self.client.query_traces(first_pass_request))
        
        logger.info(f"First pass of cost sorting request retrieved {len(first_pass_results)} traces")
        
        # Filter and sort by cost
        filtered_results = [
            t for t in first_pass_results 
            if TraceProcessor.get_cost(t, sort_by) is not None
        ]
        
        filtered_results.sort(
            key=lambda t: TraceProcessor.get_cost(t, sort_by),
            reverse=(sort_direction == "desc")
        )
        
        # Get the IDs of the top N traces
        top_ids = [t["id"] for t in filtered_results[:target_limit] if "id" in t] if target_limit else [t["id"] for t in filtered_results if "id" in t]
        
        logger.info(f"After sorting by {sort_by}, selected {len(top_ids)} trace IDs")
        
        if not top_ids:
            return []
        
        # Second pass: Fetch the full details for the selected traces
        second_pass_query = {
            "entity_name": entity_name,
            "project_name": project_name,
            "filters": {"call_ids": top_ids},
            "include_costs": include_costs,
            "include_feedback": include_feedback,
            "columns": columns,
            "expand_columns": expand_columns,
        }
        
        # Make sure we request summary if costs were requested
        if requested_synthetic_columns and "costs" in requested_synthetic_columns:
            if not columns or "summary" not in columns:
                if not second_pass_query["columns"]:
                    second_pass_query["columns"] = ["summary"]
                elif "summary" not in second_pass_query["columns"]:
                    second_pass_query["columns"].append("summary")
                logger.info("Added 'summary' to columns for cost data retrieval")
            
        second_pass_request = QueryBuilder.prepare_query_params(second_pass_query)
        second_pass_results = list(self.client.query_traces(second_pass_request))
        
        logger.info(f"Second pass retrieved {len(second_pass_results)} traces")
        
        # Add synthetic columns and invalid column warnings back to the results
        if requested_synthetic_columns or invalid_columns:
            second_pass_results = self._add_synthetic_columns(
                second_pass_results, 
                requested_synthetic_columns or [], 
                invalid_columns,
            )
        
        # Ensure the results are in the same order as the IDs
        id_to_index = {id: i for i, id in enumerate(top_ids)}
        second_pass_results.sort(
            key=lambda t: id_to_index.get(t.get("id"), float("inf"))
        )
        
        return second_pass_results 