"""
Weave MCP Server

A Model Context Protocol server for Weave traces.
"""

__version__ = "0.1.0"

# Import the functions we want to expose
from .server import mcp, cli
from .add_to_client import add_to_client_cli

# Import the original Weave client-based implementation
from .query_weave import query_traces as query_traces_client
from .query_weave import get_weave_trace_server as get_weave_trace_server_client
from .query_weave import paginated_query_traces as paginated_query_traces_client

# Import the raw HTTP-based implementation
from .query_weave_raw import query_traces, paginated_query_traces, get_weave_trace_server

# Define what gets imported with "from weave_mcp_server import *"
__all__ = [
    "mcp", 
    "cli", 
    "query_traces", 
    "paginated_query_traces",
    "get_weave_trace_server", 
    "add_to_client_cli",
    # Original client-based implementations
    "query_traces_client",
    "paginated_query_traces_client",
    "get_weave_trace_server_client",
]