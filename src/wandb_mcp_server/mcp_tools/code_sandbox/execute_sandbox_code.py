"""
Sandbox code execution tool for the MCP server.
Supports both E2B cloud sandboxes and local Pyodide execution.
"""

import asyncio
import asyncio.subprocess
import base64
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path


from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)


def check_sandbox_availability() -> tuple[bool, List[str], str]:
    """
    Check if any sandbox is available for code execution.
    
    Returns:
        tuple containing:
            - is_available (bool): Whether any sandbox is available
            - available_types (List[str]): List of available sandbox types ('e2b', 'pyodide')
            - reason (str): Explanation of availability status
    """
    # Check if sandbox is disabled
    if os.getenv("DISABLE_CODE_SANDBOX"):
        return (
            False, 
            [], 
            "Code sandbox is disabled via DISABLE_CODE_SANDBOX environment variable. "
            "Remove this variable to enable sandbox functionality."
        )
    
    available_types = []
    reasons = []
    
    # Check E2B availability
    if os.getenv("E2B_API_KEY"):
        available_types.append("e2b")
        reasons.append("E2B cloud sandbox available (API key found)")
    else:
        reasons.append("E2B not available (E2B_API_KEY not set)")
    
    # Check Pyodide/Deno availability
    try:
        result = subprocess.run(
            ["deno", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            available_types.append("pyodide")
            reasons.append("Pyodide sandbox available (Deno found)")
        else:
            reasons.append("Pyodide not available (Deno command failed)")
    except Exception as e:
        reasons.append(f"Pyodide not available (Deno not found: {str(e)})")
    
    # Determine overall availability
    is_available = len(available_types) > 0
    
    # Construct reason message
    if is_available:
        reason = f"Sandbox available. {' '.join(reasons)}"
    else:
        reason = f"No sandboxes available. {' '.join(reasons)}"
    
    return (is_available, available_types, reason)


# Configuration constants
DEFAULT_CACHE_SIZE = 100
DEFAULT_CACHE_TTL_SECONDS = 900  # 15 minutes
DEFAULT_RATE_LIMIT_REQUESTS = 100
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 300
E2B_STARTUP_TIMEOUT_SECONDS = 60

# Configuration validation functions
def validate_timeout(timeout: int, param_name: str = "timeout") -> int:
    """Validate timeout value is within acceptable range."""
    if timeout < 1:
        raise ValueError(f"{param_name} must be at least 1 second, got {timeout}")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"{param_name} must not exceed {MAX_TIMEOUT_SECONDS} seconds, got {timeout}")
    return timeout

def get_validated_env_int(env_var: str, default: int, min_val: int = 1, max_val: Optional[int] = None) -> int:
    """Get and validate integer environment variable."""
    try:
        value = int(os.getenv(env_var, str(default)))
        if value < min_val:
            logger.warning(f"{env_var}={value} is below minimum {min_val}, using {min_val}")
            return min_val
        if max_val is not None and value > max_val:
            logger.warning(f"{env_var}={value} exceeds maximum {max_val}, using {max_val}")
            return max_val
        return value
    except ValueError:
        logger.warning(f"Invalid {env_var} value, using default {default}")
        return default

# Get cache TTL from environment or use default
CACHE_TTL_SECONDS = get_validated_env_int("E2B_CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS, min_val=60, max_val=3600)

# Package installation security configuration
# Can be overridden via E2B_PACKAGE_ALLOWLIST and E2B_PACKAGE_DENYLIST env vars
DEFAULT_PACKAGE_ALLOWLIST = None  # None means allow all (unless denied)
DEFAULT_PACKAGE_DENYLIST = [
    # Packages that could be used maliciously
    "subprocess32",  # Subprocess with additional features
    "psutil",  # System and process utilities
    "pyautogui",  # GUI automation
    "pynput",  # Input control
]


def get_package_filters():
    """Get package allowlist and denylist from environment or defaults."""
    # Get allowlist
    allowlist_env = os.getenv("E2B_PACKAGE_ALLOWLIST")
    if allowlist_env:
        allowlist = [pkg.strip() for pkg in allowlist_env.split(",") if pkg.strip()]
    else:
        allowlist = DEFAULT_PACKAGE_ALLOWLIST
    
    # Get denylist
    denylist_env = os.getenv("E2B_PACKAGE_DENYLIST")
    if denylist_env:
        denylist = [pkg.strip() for pkg in denylist_env.split(",") if pkg.strip()]
    else:
        denylist = DEFAULT_PACKAGE_DENYLIST
    
    return allowlist, denylist


EXECUTE_SANDBOX_CODE_TOOL_DESCRIPTION = """
Execute Python code in a secure, isolated code sandbox environment for data analysis on queried \
Weights & Biases data. The sandbox comes with pandas and numpy pre-installed. If there is a need for data \
transforms to help answer the users' question then python code can be passed to this tool.

<usage_guidelines>
- Perfect for data analysis, visualization, and computational tasks
- Supports common packages like numpy, pandas, matplotlib
- Ideal for exploratory data analysis and quick computations
- Can be used to process W&B data safely
</usage_guidelines>

<security_features>
- **E2B**: Fully isolated cloud VM with ~150ms startup time, complete system isolation
- **Pyodide**: WebAssembly sandbox using Deno's permission model for enhanced security:
  - Explicit network permission only for package downloads
  - No filesystem access (except node_modules)
  - Process-level isolation with Deno's security sandbox
</security_features>

Args
-------
code : str
    Python code to execute in the sandbox.
timeout : int, optional
    Maximum execution time in seconds (default: 30).
install_packages : list of str, optional
    Additional packages to install for analysis on top of numpy and pandas.
    - E2B sandbox: Supports dynamic package installation with security filters
      (configurable via E2B_PACKAGE_ALLOWLIST and E2B_PACKAGE_DENYLIST env vars)
    - Pyodide sandbox: Pre-loaded with numpy, pandas, matplotlib only. 
      Additional pure Python packages can be imported but not installed.

Returns
-------
dict
    Dictionary with the following keys:
    success : bool
        Whether execution succeeded.
    output : str
        Standard output from code execution.
    error : str
        Error message if execution failed.
    logs : list of str
        Execution logs.
    sandbox_used : str
        Type of sandbox that was used.

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


class ExecutionCache:
    """Simple LRU cache for code execution results."""
    
    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _get_cache_key(self, code: str, sandbox_type: str, packages: Optional[List[str]]) -> str:
        """Generate cache key from code and parameters."""
        package_str = ",".join(sorted(packages)) if packages else ""
        content = f"{code}|{sandbox_type}|{package_str}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, code: str, sandbox_type: str, packages: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired."""
        key = self._get_cache_key(code, sandbox_type, packages)
        if key in self.cache:
            entry, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                # Move to end to maintain LRU order
                self.cache.move_to_end(key)
                logger.debug(f"Cache hit for key {key[:8]}...")
                return entry
            else:
                # Expired
                del self.cache[key]
        return None
    
    def set(self, code: str, sandbox_type: str, result: Dict[str, Any], packages: Optional[List[str]] = None):
        """Cache execution result."""
        key = self._get_cache_key(code, sandbox_type, packages)
        self.cache[key] = (result, datetime.now())
        self.cache.move_to_end(key)
        
        # Evict oldest if over size limit
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)




# Removed E2BSandboxPool - we'll use a single persistent sandbox instead


class E2BSandbox:
    """E2B cloud sandbox implementation with a single persistent instance."""
    
    # Class-level persistent sandbox instance
    _shared_sandbox = None
    _sandbox_lock = asyncio.Lock()
    _api_key = None
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sandbox = None  # Initialize the sandbox attribute
        E2BSandbox._api_key = api_key
    
    async def __aenter__(self):
        """Context manager entry - get sandbox reference."""
        await self.create_sandbox()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release sandbox reference."""
        await self.close_sandbox()
        return False
    
    @classmethod
    async def get_or_create_sandbox(cls):
        """Get the shared sandbox instance, creating it if necessary."""
        async with cls._sandbox_lock:
            if cls._shared_sandbox is None:
                if not cls._api_key:
                    raise ValueError("E2B API key not set")
                logger.info("Creating new E2B sandbox instance")
                from e2b_code_interpreter import AsyncSandbox
                
                os.environ["E2B_API_KEY"] = cls._api_key
                
                # Get timeout from environment or use E2B default (15 minutes)
                # E2B expects timeout in seconds, not milliseconds
                timeout_seconds = get_validated_env_int(
                    "E2B_SANDBOX_TIMEOUT_SECONDS", 
                    900,  # 15 minutes default
                    min_val=60,  # 1 minute minimum
                    max_val=3600  # 1 hour maximum
                )
                logger.info(f"Creating E2B sandbox with timeout: {timeout_seconds}s ({timeout_seconds/60:.1f} minutes)")
                
                cls._shared_sandbox = await AsyncSandbox.create(timeout=timeout_seconds)
                logger.info("E2B sandbox instance created successfully")
            return cls._shared_sandbox
    
    async def create_sandbox(self):
        """Get reference to the shared sandbox instance."""
        try:
            self.sandbox = await self.get_or_create_sandbox()
            logger.debug("Using shared E2B sandbox instance")
        except Exception as e:
            logger.error(f"Failed to create/get E2B sandbox: {e}", exc_info=True)
            raise
    
    async def install_packages(self, packages: List[str]) -> bool:
        """Install packages in the sandbox."""
        if not packages or not self.sandbox:
            return True
        
        try:
            # Get package filters
            allowlist, denylist = get_package_filters()
            
            # Sanitize and filter package names
            filtered_packages = []
            denied_packages = []
            
            for pkg in packages:
                pkg = pkg.strip()
                # First check format safety
                if not re.match(r'^[a-zA-Z0-9\-_.]+$', pkg):
                    logger.warning(f"Package '{pkg}' filtered due to invalid characters")
                    continue
                
                # Check against denylist
                if denylist and pkg.lower() in [d.lower() for d in denylist]:
                    denied_packages.append(pkg)
                    continue
                
                # Check against allowlist (if specified)
                if allowlist and pkg.lower() not in [a.lower() for a in allowlist]:
                    denied_packages.append(pkg)
                    continue
                
                filtered_packages.append(pkg)
            
            if denied_packages:
                logger.warning(f"Denied packages: {', '.join(denied_packages)}")
            
            if not filtered_packages:
                if denied_packages:
                    return False  # All packages were denied
                return True  # No packages to install
            
            # Install packages
            package_str = " ".join(filtered_packages)
            logger.info(f"Installing packages: {package_str}")
            
            result = await self.sandbox.commands.run(
                f"uv pip install --no-cache-dir {package_str}",
                timeout=E2B_STARTUP_TIMEOUT_SECONDS
            )
            
            success = result.exit_code == 0
            if not success:
                logger.error(f"Package installation failed: {result.stderr}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to install packages: {e}")
            return False
    
    async def execute_code(self, code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS, install_packages_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute Python code in the E2B sandbox."""
        # Validate timeout
        timeout = validate_timeout(timeout)
        
        logger.debug(f"execute_code called, self.sandbox is: {self.sandbox}")
        if not hasattr(self, 'sandbox') or self.sandbox is None:
            logger.debug("self.sandbox not set, calling create_sandbox()")
            await self.create_sandbox()
            logger.debug(f"After create_sandbox, self.sandbox is: {self.sandbox}")
        
        # Install packages if requested
        if install_packages_list:
            success = await self.install_packages(install_packages_list)
            if not success:
                return {
                    "success": False,
                    "output": "",
                    "error": "Failed to install requested packages",
                    "logs": [],
                }
        
        try:
            # Write code to a temporary file in the sandbox to avoid shell escaping issues
            file_path = "/tmp/code_to_execute.py"
            
            # Write the code to file
            await self.sandbox.files.write(file_path, code)
            
            # Execute the file
            execution = await self.sandbox.commands.run(
                f"python {file_path}",
                timeout=timeout
            )
            
            # Format the result
            output = execution.stdout if execution.stdout else ""
            error_output = execution.stderr if execution.stderr else ""
            
            success = execution.exit_code == 0
            error_msg = error_output if error_output and not success else None
            
            # Clean up the temporary file
            try:
                await self.sandbox.commands.run(f"rm {file_path}")
            except Exception:
                pass  # Ignore cleanup errors
            
            return {
                "success": success,
                "output": output,
                "error": error_msg,
                "logs": [output] if not error_output else [output, error_output],
            }
            
        except Exception as e:
            logger.error(f"E2B execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "output": "",
                "error": f"E2B execution failed: {str(e)}",
                "logs": [],
            }
    
    async def close_sandbox(self):
        """Release the reference to the sandbox (but keep it running)."""
        # Don't actually close the shared sandbox - just release our reference
        self.sandbox = None
        logger.debug("Released reference to shared E2B sandbox")
    
    async def writeFile(self, path: str, content: str) -> None:
        """Write a file to the E2B sandbox using native file operations."""
        if not self.sandbox:
            await self.create_sandbox()
        
        try:
            await self.sandbox.files.write(path, content)
            logger.debug(f"Wrote file to E2B sandbox: {path}")
        except Exception as e:
            raise SandboxError(f"Failed to write file to E2B sandbox: {e}")
    
    @classmethod
    async def cleanup_shared_sandbox(cls):
        """Explicitly close the shared sandbox instance (for cleanup)."""
        async with cls._sandbox_lock:
            if cls._shared_sandbox is not None:
                try:
                    await cls._shared_sandbox.close()
                    logger.info("Closed shared E2B sandbox instance")
                except Exception as e:
                    logger.error(f"Error closing shared E2B sandbox: {e}")
                finally:
                    cls._shared_sandbox = None


class PyodideSandbox:
    """Local Pyodide sandbox implementation using Deno for enhanced security."""
    
    # Class-level persistent process
    _shared_process = None
    _process_lock = asyncio.Lock()
    _initialized = False
    
    def __init__(self):
        self.available = self._check_deno_available()
        self._pyodide_script_path = Path(__file__).parent / "pyodide_sandbox.ts"
    
    async def __aenter__(self):
        """Context manager entry - ensure process is ready."""
        if self.available:
            await self.get_or_create_process(self._pyodide_script_path)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - process cleanup handled at shutdown."""
        # Don't cleanup process here as it's shared across executions
        return False
    
    def _check_deno_available(self) -> bool:
        """Check if Deno is available."""
        try:
            result = subprocess.run(
                ["deno", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    @classmethod
    async def get_or_create_process(cls, script_path: Path):
        """Get or create the shared Pyodide process."""
        async with cls._process_lock:
            if cls._shared_process is None or cls._shared_process.returncode is not None:
                logger.info("Starting persistent Pyodide sandbox process")
                cls._shared_process = await asyncio.create_subprocess_exec(
                    'deno', 'run',
                    '--allow-net',  # For downloading Pyodide and packages
                    '--allow-read',  # To read local files
                    '--allow-write',  # To write output files and cache
                    '--allow-env',  # For environment variables
                    str(script_path),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                cls._initialized = True
                
                # Read initialization messages
                try:
                    # Wait for "ready" message with timeout
                    ready_task = asyncio.create_task(cls._read_until_ready())
                    await asyncio.wait_for(ready_task, timeout=60)
                    logger.info("Pyodide sandbox process initialized")
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for Pyodide to initialize")
                    if cls._shared_process:
                        cls._shared_process.kill()
                        await cls._shared_process.wait()
                    cls._shared_process = None
                    raise SandboxError(
                        "Failed to initialize Pyodide sandbox. "
                        "Ensure Deno is properly installed and has network access to download Pyodide. "
                        "Try running: deno --version"
                    )
                    
            return cls._shared_process
    
    @classmethod
    async def _read_until_ready(cls):
        """Read stderr until we see the ready message."""
        if not cls._shared_process:
            return
            
        while True:
            line = await cls._shared_process.stderr.readline()
            if not line:
                break
            line_str = line.decode('utf-8').strip()
            logger.debug(f"Pyodide init: {line_str}")
            if "Pyodide sandbox server ready" in line_str:
                break
    
    async def execute_code(self, code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
        """Execute Python code using our persistent Pyodide process."""
        # Validate timeout
        timeout = validate_timeout(timeout)
        
        if not self.available:
            raise SandboxError(
                "Deno is not available for Pyodide execution. "
                "Install Deno with: curl -fsSL https://deno.land/install.sh | sh"
            )
        
        if not self._pyodide_script_path.exists():
            raise SandboxError(f"Pyodide script not found at {self._pyodide_script_path}")
        
        try:
            # Get or create the persistent process
            process = await self.get_or_create_process(self._pyodide_script_path)
            
            # Create execution request
            request = {
                "type": "execute",
                "code": code,
                "timeout": timeout
            }
            
            # Send request as a single line JSON
            request_json = json.dumps(request) + '\n'
            process.stdin.write(request_json.encode())
            await process.stdin.drain()
            
            # Read response with timeout
            try:
                response_line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=timeout + 5  # Give extra time for process overhead
                )
            except asyncio.TimeoutError:
                # Don't kill the process on timeout, just return error
                return {
                    "success": False,
                    "output": "",
                    "error": f"Execution timed out after {timeout} seconds",
                    "logs": [],
                }
            
            # Parse the response
            try:
                if response_line:
                    result = json.loads(response_line.decode('utf-8'))
                    return result
                else:
                    return {
                        "success": False,
                        "output": "",
                        "error": "No response from Pyodide process",
                        "logs": [],
                    }
                    
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Failed to parse response: {e}",
                    "logs": [],
                }
            
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Pyodide execution failed: {str(e)}",
                "logs": [],
            }
    
    @classmethod
    async def cleanup_shared_process(cls):
        """Explicitly close the shared Pyodide process (for cleanup)."""
        async with cls._process_lock:
            if cls._shared_process is not None:
                try:
                    # First try graceful termination
                    cls._shared_process.terminate()
                    try:
                        # Wait for process to exit with timeout
                        await asyncio.wait_for(cls._shared_process.wait(), timeout=5.0)
                        logger.info("Pyodide process terminated gracefully")
                    except asyncio.TimeoutError:
                        # Force kill if termination times out
                        logger.warning("Pyodide process did not terminate gracefully, forcing kill")
                        cls._shared_process.kill()
                        await cls._shared_process.wait()
                        logger.info("Pyodide process killed forcefully")
                except Exception as e:
                    logger.error(f"Error closing shared Pyodide process: {e}")
                finally:
                    cls._shared_process = None
                    cls._initialized = False
    
    async def writeFile(self, path: str, content: str) -> None:
        """Write a file to the Pyodide sandbox using persistent process."""
        if not self.available:
            raise SandboxError(
                "Deno is not available for Pyodide execution. "
                "Install Deno with: curl -fsSL https://deno.land/install.sh | sh"
            )
        
        try:
            # Get or create the persistent process
            process = await self.get_or_create_process(self._pyodide_script_path)
            
            # Create file write request
            request = {
                "type": "writeFile",
                "path": path,
                "content": content
            }
            
            # Send request as a single line JSON
            request_json = json.dumps(request) + '\n'
            process.stdin.write(request_json.encode())
            await process.stdin.drain()
            
            # Read response with timeout
            try:
                response_line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=10
                )
            except asyncio.TimeoutError:
                raise SandboxError(f"Timeout writing file to {path}")
            
            # Parse the response
            if response_line:
                result = json.loads(response_line.decode('utf-8'))
                if not result.get("success", False):
                    raise SandboxError(f"Failed to write file: {result.get('error', 'Unknown error')}")
            else:
                raise SandboxError("No response from Pyodide process")
                
        except Exception as e:
            if isinstance(e, SandboxError):
                raise
            raise SandboxError(f"Failed to write file to Pyodide: {str(e)}")
    




# Global cache instance
_execution_cache = ExecutionCache(ttl_seconds=CACHE_TTL_SECONDS)

# Get rate limit configuration from environment
RATE_LIMIT_REQUESTS = get_validated_env_int(
    "SANDBOX_RATE_LIMIT_REQUESTS", 
    DEFAULT_RATE_LIMIT_REQUESTS,
    min_val=1,
    max_val=1000
)
RATE_LIMIT_WINDOW_SECONDS = get_validated_env_int(
    "SANDBOX_RATE_LIMIT_WINDOW_SECONDS",
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    min_val=1,
    max_val=300  # 5 minutes max
)

# Rate limiting
class RateLimiter:
    """Simple rate limiter for sandbox executions."""
    
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self._lock = asyncio.Lock()
        logger.info(f"Rate limiter configured: {max_requests} requests per {window_seconds} seconds")
    
    async def check_rate_limit(self) -> bool:
        """Check if request is within rate limit."""
        async with self._lock:
            now = time.time()
            # Remove old requests
            self.requests = [r for r in self.requests if now - r < self.window_seconds]
            
            if len(self.requests) >= self.max_requests:
                return False
            
            self.requests.append(now)
            return True


_rate_limiter = RateLimiter()


async def execute_sandbox_code(
    code: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sandbox_type: Optional[str] = None,
    install_packages: Optional[List[str]] = None,
    return_sandbox: bool = False,
) -> Dict[str, Any]:
    """
    Execute Python code in a secure sandbox environment.
    
    Automatically selects the best available sandbox or uses the specified type.
    """
    start_time = time.time()
    
    # Check rate limit
    if not await _rate_limiter.check_rate_limit():
        return {
            "success": False,
            "output": "",
            "error": "Rate limit exceeded. Please try again later.",
            "logs": [],
            "sandbox_used": "none",
        }
    
    # Validate timeout
    try:
        timeout = validate_timeout(timeout)
    except ValueError as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "logs": [],
            "sandbox_used": "none",
        }
    
    # Validate sandbox type if specified
    valid_sandbox_types = ["e2b", "pyodide", "auto", None]
    if sandbox_type and sandbox_type not in valid_sandbox_types:
        return {
            "success": False,
            "output": "",
            "error": f"Invalid sandbox_type: '{sandbox_type}'. Must be one of: {', '.join(str(t) for t in valid_sandbox_types[:-1])}",
            "logs": [],
            "sandbox_used": "none",
        }
    
    # Normalize "auto" to None for auto-selection
    if sandbox_type == "auto":
        sandbox_type = None
    
    # Determine which sandbox to use
    sandboxes_to_try = []
    
    if sandbox_type:
        # User specified a sandbox type
        if sandbox_type == "e2b":
            api_key = os.getenv("E2B_API_KEY")
            if api_key is not None:
                sandboxes_to_try.append(("e2b", E2BSandbox(api_key)))
        elif sandbox_type == "pyodide":
            pyodide = PyodideSandbox()
            if pyodide.available:
                sandboxes_to_try.append(("pyodide", pyodide))
    else:
        # Auto-select based on availability
        # Check cache first
        for sb_type in ["e2b", "pyodide"]:
            cached_result = _execution_cache.get(code, sb_type, install_packages)
            if cached_result:
                # Add execution time to cached result
                cached_result["execution_time_ms"] = 0  # Cached, so no execution time
                return cached_result
        
        # Try E2B first if available
        api_key = os.getenv("E2B_API_KEY")
        if api_key is not None:
            sandboxes_to_try.append(("e2b", E2BSandbox(api_key)))
        
        # Then try Pyodide
        pyodide = PyodideSandbox()
        if pyodide.available:
            sandboxes_to_try.append(("pyodide", pyodide))
        
    
    # Check if we have any sandboxes available
    if not sandboxes_to_try:
        return {
            "success": False,
            "output": "",
            "error": (
                "No sandboxes available. To enable code execution:\n"
                "1. For cloud sandbox: Set E2B_API_KEY environment variable (get key at https://e2b.dev)\n"
                "2. For local sandbox: Install Deno with: curl -fsSL https://deno.land/install.sh | sh"
            ),
            "logs": [],
            "sandbox_used": "none",
            "execution_time_ms": int((time.time() - start_time) * 1000),
        }
    
    # Try each sandbox in order
    last_error = None
    for sandbox_name, sandbox in sandboxes_to_try:
        try:
            logger.info(f"Attempting to execute code in {sandbox_name} sandbox")
            
            # Execute based on sandbox type
            if sandbox_name == "e2b":
                result = await sandbox.execute_code(code, timeout, install_packages)
                await sandbox.close_sandbox()
            else:
                # Pyodide doesn't support dynamic package installation
                if install_packages and sandbox_name == "pyodide":
                    logger.warning(
                        "Pyodide sandbox doesn't support dynamic package installation. "
                        "Pre-loaded packages: numpy, pandas, matplotlib. "
                        "Additional packages can be imported if they're pure Python."
                    )
                result = await sandbox.execute_code(code, timeout)
            
            # Add sandbox info and execution time
            result["sandbox_used"] = sandbox_name
            result["execution_time_ms"] = int((time.time() - start_time) * 1000)
            
            # Optionally include sandbox instance for file operations
            if return_sandbox:
                result["_sandbox_instance"] = sandbox
                result["_sandbox_type"] = sandbox_name
            
            # Cache successful results
            if result["success"]:
                _execution_cache.set(code, sandbox_name, result, install_packages)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute in {sandbox_name} sandbox: {e}")
            last_error = str(e)
            continue
    
    # All sandboxes failed
    return {
        "success": False,
        "output": "",
        "error": f"All sandbox types failed. Last error: {last_error}",
        "logs": [],
        "sandbox_used": "none",
        "execution_time_ms": int((time.time() - start_time) * 1000),
    }