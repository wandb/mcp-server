"""
Utility functions for writing files to sandbox environments.
Provides a clean interface for writing data to both E2B and Pyodide sandboxes.
"""

import json
import logging
from typing import Any, Dict, Optional, Union

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)


async def write_json_to_sandbox(
    json_data: Union[str, Dict[str, Any]],
    filename: str,
    path_prefix: str = "/tmp/",
) -> None:
    """
    Write JSON data to a file in any available sandbox using native file operations.
    
    This is a fire-and-forget operation that writes to the sandbox if available.
    It's designed to be called from server.py without needing to manage sandbox instances.
    
    Args:
        json_data: JSON data as string or dict
        filename: Name of the file to create
        path_prefix: Directory prefix for the file (default: /tmp/)
    """
    try:
        # Import here to avoid circular imports
        from wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code import (
            check_sandbox_availability,
            E2BSandbox,
            PyodideSandbox,
        )
        
        # Check if sandbox is available
        available, sandbox_types, _ = check_sandbox_availability()
        if not available:
            logger.debug("No sandbox available for file writing")
            return
        
        # Convert dict to JSON string if needed
        if isinstance(json_data, dict):
            content = json.dumps(json_data, indent=2)
        else:
            content = str(json_data)
        
        # Ensure path_prefix ends with /
        if not path_prefix.endswith('/'):
            path_prefix += '/'
        
        full_path = f"{path_prefix}{filename}"
        
        # Try to write using available sandbox
        if "e2b" in sandbox_types:
            # Use E2B
            import os
            api_key = os.getenv("E2B_API_KEY")
            if api_key:
                sandbox = E2BSandbox(api_key)
                await sandbox.create_sandbox()
                await sandbox.writeFile(full_path, content)
                await sandbox.close_sandbox()  # Just releases the reference, doesn't close the sandbox
                logger.info(f"Wrote {filename} to E2B sandbox")
                return
        
        if "pyodide" in sandbox_types:
            # Use Pyodide
            sandbox = PyodideSandbox()
            await sandbox.writeFile(full_path, content)
            logger.info(f"Wrote {filename} to Pyodide sandbox")
            return
            
    except Exception as e:
        logger.error(f"Error writing {filename} to sandbox: {e}", exc_info=True)
        # Don't raise - this is a best-effort operation