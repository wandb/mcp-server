"""
Weave MCP Server

A Model Context Protocol server for Weave traces.
"""

__version__ = "0.1.0"

# Import the functions we want to expose
from .server import mcp, cli
from .query import query_traces, get_weave_trace_server

# Define what gets imported with "from weave_mcp_server import *"
__all__ = ["mcp", "cli", "query_traces", "get_weave_trace_server"] 