#!/usr/bin/env python3
"""
Example script demonstrating how to use the raw HTTP implementation of the Weave API client.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any

# Import the raw HTTP implementation
from wandb_mcp_server import query_traces, paginated_query_traces

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Example script for the raw HTTP Weave API client")
    parser.add_argument("--entity", required=True, help="W&B entity name")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--run-id", help="W&B run ID to filter by")
    parser.add_argument("--op-name", help="Weave op name to filter by")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    parser.add_argument("--api-key", help="W&B API key")
    args = parser.parse_args()
    
    # Set the API key from the command line or environment
    api_key = args.api_key or os.environ.get("WANDB_API_KEY")
    if not api_key:
        logger.warning("No W&B API key provided. Using anonymous access.")
    
    # Set up the filters
    filters = {"trace_roots_only": True}
    
    if args.op_name:
        filters["op_name"] = args.op_name
    
    if args.run_id:
        filters["wb_run_id"] = {"$contains": args.run_id}
    
    # Query traces
    logger.info(f"Querying traces for {args.entity}/{args.project} with filters: {filters}")
    traces = query_traces(
        entity_name=args.entity,
        project_name=args.project,
        filters=filters,
        sort_by="started_at",
        sort_direction="desc",
        limit=args.limit,
        include_costs=True,
        include_feedback=True,
        api_key=api_key,
    )
    
    # Print the results
    if traces:
        logger.info(f"Found {len(traces)} traces")
        for i, trace in enumerate(traces):
            print(f"\n--- Trace {i+1} ---")
            print(f"ID: {trace['id']}")
            print(f"Op Name: {trace['op_name']}")
            print(f"Display Name: {trace['display_name']}")
            print(f"Started At: {trace['started_at']}")
            print(f"Ended At: {trace['ended_at']}")
            print(f"Run ID: {trace.get('wb_run_id', 'N/A')}")
            
            # Check for costs
            if trace.get('summary') and trace['summary'].get('weave') and trace['summary']['weave'].get('costs'):
                print("\nCosts:")
                print(json.dumps(trace['summary']['weave']['costs'], indent=2))
            else:
                print("\nNo costs found")
    else:
        logger.info("No traces found")
    
    # Demonstrate paginated query
    logger.info(f"Performing paginated query for {args.entity}/{args.project}")
    import asyncio
    
    async def run_paginated_query():
        result = await paginated_query_traces(
            entity_name=args.entity,
            project_name=args.project,
            chunk_size=2,  # Small chunk size for demonstration
            filters=filters,
            target_limit=args.limit,
            include_costs=True,
            include_feedback=True,
            api_key=api_key,
        )
        
        print("\n--- Paginated Query Results ---")
        print(f"Total Traces: {result['metadata']['total_traces']}")
        print(f"Time Range: {result['metadata']['time_range']}")
        print(f"Status Summary: {result['metadata']['status_summary']}")
        print(f"Op Distribution: {result['metadata']['op_distribution']}")
        
        print("\nTraces:")
        for i, trace in enumerate(result['traces']):
            print(f"\n--- Trace {i+1} ---")
            print(f"ID: {trace['id']}")
            print(f"Op Name: {trace['op_name']}")
            print(f"Display Name: {trace['display_name']}")
    
    # Run the async function
    asyncio.run(run_paginated_query())

if __name__ == "__main__":
    main()