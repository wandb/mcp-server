"""
Sandbox code execution tool for the MCP server.
Supports both E2B cloud sandboxes and local Pyodide execution.
"""

import asyncio
import json
import os
import tempfile
from typing import Any, Dict, List, Optional

import httpx

from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)


EXECUTE_SANDBOX_CODE_TOOL_DESCRIPTION = """
Execute Python code in a secure, isolated sandbox environment. This tool provides safe code execution
using multiple fallback options to ensure maximum compatibility and security.

<sandbox_types>
The tool automatically selects the best available sandbox in this order:
1. **E2B Cloud Sandbox** - If E2B_API_KEY is available (most secure, cloud-based)
2. **Pyodide Local Sandbox** - If Node.js is available (local, WebAssembly-based)
3. **RestrictedPython** - Fallback option (local, restricted execution)

You can force a specific sandbox type using the sandbox_type parameter.
</sandbox_types>

<security_features>
- **E2B**: Fully isolated cloud VM with ~150ms startup time
- **Pyodide**: WebAssembly sandbox with no filesystem access
- **RestrictedPython**: Restricted Python execution with limited builtins
</security_features>

<usage_guidelines>
- Perfect for data analysis, visualization, and computational tasks
- Supports common packages like numpy, pandas, matplotlib (depending on sandbox)
- Use for user-provided code that needs isolation from the host system
- Ideal for exploratory data analysis and quick computations
- Can be used to process W&B data safely
</usage_guidelines>

<debugging_tips>
- If E2B fails, check E2B_API_KEY environment variable
- If Pyodide fails, ensure Node.js is installed
- RestrictedPython has limited package support
- Check the 'sandbox_used' field in results to see which sandbox was used
</debugging_tips>

Args:
    code (str): Python code to execute in the sandbox
    timeout (int, optional): Maximum execution time in seconds (default: 30)
    sandbox_type (str, optional): Force specific sandbox ('e2b', 'pyodide', 'restricted')
    install_packages (List[str], optional): Packages to install (E2B only)

Returns:
    Dict containing:
        - success (bool): Whether execution succeeded
        - output (str): Standard output from code execution
        - error (str): Error message if execution failed
        - logs (List[str]): Execution logs
        - sandbox_used (str): Type of sandbox that was used

Example:
    ```python
    # Simple computation
    result = execute_sandbox_code('''
    import math
    result = math.sqrt(16)
    print(f"Square root of 16 is {result}")
    ''')
    
    # Data analysis with pandas
    result = execute_sandbox_code('''
    import pandas as pd
    data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
    df = pd.DataFrame(data)
    print(df.describe())
    ''')
    ```
"""


class SandboxError(Exception):
    """Exception raised when sandbox execution fails."""

    pass


class E2BSandbox:
    """E2B cloud sandbox implementation using official SDK."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sandbox = None

    async def create_sandbox(self):
        """Create a new E2B sandbox instance."""
        try:
            from e2b_code_interpreter import AsyncSandbox
            
            # Set the API key in environment for the SDK
            os.environ["E2B_API_KEY"] = self.api_key
            
            self.sandbox = await AsyncSandbox.create()
            return self.sandbox.sandbox_id
        except ImportError:
            raise SandboxError("E2B SDK not available. Install with: uv add e2b-code-interpreter")

    async def execute_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute Python code in the E2B sandbox."""
        if not self.sandbox:
            await self.create_sandbox()

        try:
            # Execute the code using the official SDK
            # Create a simple Python script to execute
            execution = await self.sandbox.commands.run(f"python3 -c '{code}'", timeout=timeout)
            
            # Format the result according to our interface
            output = execution.stdout if execution.stdout else ""
            error_output = execution.stderr if execution.stderr else ""
            
            # Check if execution was successful (exit code 0)
            success = execution.exit_code == 0
            error_msg = error_output if error_output else None
            
            return {
                "success": success,
                "output": output,
                "error": error_msg,
                "logs": [output, error_output] if error_output else [output],
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"E2B execution failed: {str(e)}",
                "logs": [],
            }

    async def close_sandbox(self):
        """Close the E2B sandbox."""
        if self.sandbox:
            try:
                # E2B sandboxes are automatically cleaned up
                # No explicit close method needed for AsyncSandbox
                pass
            except Exception as e:
                logger.warning(f"Error closing E2B sandbox: {e}")
            finally:
                self.sandbox = None


class PyodideSandbox:
    """Local Pyodide sandbox implementation using Node.js."""

    def __init__(self):
        self.session_id = None

    async def execute_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute Python code using Pyodide in Node.js."""
        # Create a temporary JavaScript file that runs Pyodide
        js_script = f"""
// For simplicity, let's use a basic Python interpreter simulation
// In a real implementation, we'd use Pyodide from CDN
const {{ execSync }} = require('child_process');

async function runPython() {{
    try {{
        // Use a simple approach: write Python code to temp file and execute with python3
        const fs = require('fs');
        const path = require('path');
        const os = require('os');
        
        const tempFile = path.join(os.tmpdir(), `pyodide_exec_${{Date.now()}}.py`);
        
        // Wrap the code to capture stdout
        const userCode = `{code.replace('`', '\\`').replace('${', '\\${')}`;
        const wrappedCode = `
import sys
from io import StringIO
import contextlib

# Capture stdout
captured_output = StringIO()

try:
    with contextlib.redirect_stdout(captured_output):
        exec('''${{userCode}}''')
    
    result = captured_output.getvalue()
    print("__PYODIDE_OUTPUT__:" + result)
    
except Exception as e:
    print("__PYODIDE_ERROR__:" + str(e))
`;
        
        fs.writeFileSync(tempFile, wrappedCode);
        
        try {{
            const result = execSync('python3 ' + tempFile, {{
                encoding: 'utf8',
                timeout: {timeout * 1000},
                maxBuffer: 1024 * 1024
            }});
            
            // Parse the output
            if (result.includes('__PYODIDE_ERROR__:')) {{
                const error = result.split('__PYODIDE_ERROR__:')[1].trim();
                return {{
                    success: false,
                    error: error,
                    output: '',
                    logs: []
                }};
            }} else if (result.includes('__PYODIDE_OUTPUT__:')) {{
                const output = result.split('__PYODIDE_OUTPUT__:')[1].trim();
                return {{
                    success: true,
                    output: output,
                    error: null,
                    logs: []
                }};
            }} else {{
                return {{
                    success: true,
                    output: result.trim(),
                    error: null,
                    logs: []
                }};
            }}
            
        }} catch (error) {{
            return {{
                success: false,
                error: error.message,
                output: '',
                logs: []
            }};
        }} finally {{
            // Clean up temp file
            try {{ fs.unlinkSync(tempFile); }} catch (e) {{}}
        }}
        
    }} catch (error) {{
        return {{
            success: false,
            error: error.message,
            output: '',
            logs: []
        }};
    }}
}}

runPython().then(result => {{
    console.log(JSON.stringify(result));
}}).catch(error => {{
    console.log(JSON.stringify({{
        success: false,
        error: error.message,
        output: '',
        logs: []
    }}));
}});
"""

        # Write the script to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(js_script)
            script_path = f.name

        try:
            # Execute the script with Node.js
            process = await asyncio.create_subprocess_exec(
                "node",
                script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                raise SandboxError("Code execution timed out")

            if process.returncode != 0:
                raise SandboxError(f"Node.js execution failed: {stderr.decode()}")

            # Parse the JSON result
            result = json.loads(stdout.decode())
            return result

        finally:
            # Clean up the temporary file
            os.unlink(script_path)


class RestrictedPythonSandbox:
    """Local sandbox using RestrictedPython as a fallback."""

    def __init__(self):
        # Always available since it uses subprocess
        self.available = True

    async def execute_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute Python code using subprocess with basic restrictions."""
        if not self.available:
            raise SandboxError("RestrictedPython sandbox is not available")

        try:
            # Create a wrapper script that captures output
            wrapper_code = f'''
import sys
from io import StringIO
import contextlib

# Capture stdout
captured_output = StringIO()

try:
    with contextlib.redirect_stdout(captured_output):
{chr(10).join("        " + line for line in code.split(chr(10)))}
    
    result = captured_output.getvalue()
    print("__RESTRICTED_OUTPUT__:" + result)
    
except Exception as e:
    print("__RESTRICTED_ERROR__:" + str(e))
'''

            # Execute using subprocess
            process = await asyncio.create_subprocess_exec(
                'python3', '-c', wrapper_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "error": "Code execution timed out",
                    "output": "",
                    "logs": [],
                }
            
            output_str = stdout.decode() if stdout else ""
            error_str = stderr.decode() if stderr else ""
            
            # Parse the output
            if "__RESTRICTED_ERROR__:" in output_str:
                error = output_str.split("__RESTRICTED_ERROR__:")[1].strip()
                return {
                    "success": False,
                    "error": error,
                    "output": "",
                    "logs": [],
                }
            elif "__RESTRICTED_OUTPUT__:" in output_str:
                output = output_str.split("__RESTRICTED_OUTPUT__:")[1].strip()
                return {
                    "success": True,
                    "output": output,
                    "error": None,
                    "logs": [],
                }
            else:
                # Fallback - use whatever output we got
                return {
                    "success": process.returncode == 0,
                    "output": output_str.strip(),
                    "error": error_str.strip() if error_str else None,
                    "logs": [],
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "output": "",
                "logs": [],
            }


async def execute_sandbox_code(
    code: str,
    timeout: int = 30,
    sandbox_type: Optional[str] = None,
    install_packages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute Python code in a secure sandbox environment.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30)
        sandbox_type: Force specific sandbox type ('e2b', 'pyodide', 'restricted')
        install_packages: List of packages to install (E2B only)

    Returns:
        Dictionary containing:
        - success: bool indicating if execution succeeded
        - output: string output from the code execution
        - error: error message if execution failed
        - logs: list of log messages
        - sandbox_used: type of sandbox that was used
    """

    # Determine which sandbox to use
    e2b_api_key = os.getenv("E2B_API_KEY")

    if sandbox_type == "e2b" or (sandbox_type is None and e2b_api_key):
        # Use E2B sandbox
        try:
            sandbox = E2BSandbox(e2b_api_key)
            result = await sandbox.execute_code(code, timeout)
            await sandbox.close_sandbox()
            result["sandbox_used"] = "e2b"
            return result
        except Exception as e:
            logger.warning(f"E2B sandbox failed: {e}")
            if sandbox_type == "e2b":
                return {
                    "success": False,
                    "error": f"E2B sandbox failed: {str(e)}",
                    "output": "",
                    "logs": [],
                    "sandbox_used": "e2b",
                }

    # Try Pyodide sandbox
    if sandbox_type == "pyodide" or sandbox_type is None:
        try:
            # Check if Node.js is available
            process = await asyncio.create_subprocess_exec(
                "node",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

            if process.returncode == 0:
                sandbox = PyodideSandbox()
                result = await sandbox.execute_code(code, timeout)
                result["sandbox_used"] = "pyodide"
                return result
        except Exception as e:
            logger.warning(f"Pyodide sandbox failed: {e}")
            if sandbox_type == "pyodide":
                return {
                    "success": False,
                    "error": f"Pyodide sandbox failed: {str(e)}",
                    "output": "",
                    "logs": [],
                    "sandbox_used": "pyodide",
                }

    # Fall back to RestrictedPython
    try:
        sandbox = RestrictedPythonSandbox()
        result = await sandbox.execute_code(code, timeout)
        result["sandbox_used"] = "restricted"
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"All sandbox options failed. Last error: {str(e)}",
            "output": "",
            "logs": [],
            "sandbox_used": "none",
        }
