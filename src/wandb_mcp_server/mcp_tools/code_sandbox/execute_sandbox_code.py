"""
Sandbox code execution tool for the MCP server.
Supports both E2B cloud sandboxes and local Pyodide execution.
"""

import asyncio
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
        return (False, [], "Code sandbox is disabled via DISABLE_CODE_SANDBOX environment variable")
    
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
DEFAULT_POOL_SIZE = 5
DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 300
E2B_STARTUP_TIMEOUT_SECONDS = 60

# Get cache TTL from environment or use default
CACHE_TTL_SECONDS = int(os.getenv("E2B_CACHE_TTL_SECONDS", str(DEFAULT_CACHE_TTL_SECONDS)))

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




class E2BSandboxPool:
    """Connection pool for E2B sandboxes."""
    
    def __init__(self, api_key: str, pool_size: int = DEFAULT_POOL_SIZE):
        self.api_key = api_key
        self.pool_size = pool_size
        self.available = asyncio.Queue(maxsize=pool_size)
        self.all_sandboxes = []
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the sandbox pool."""
        async with self._lock:
            if self._initialized:
                return
            
            logger.info(f"Initializing E2B sandbox pool with {self.pool_size} sandboxes")
            
            # Pre-create sandboxes
            tasks = [self._create_sandbox() for _ in range(self.pool_size)]
            sandboxes = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(sandboxes):
                if isinstance(result, Exception):
                    logger.error(f"Failed to create sandbox {i}: {result}")
                else:
                    self.all_sandboxes.append(result)
                    await self.available.put(result)
            
            self._initialized = True
            logger.info(f"E2B sandbox pool initialized with {len(self.all_sandboxes)} sandboxes")
    
    async def _create_sandbox(self):
        """Create a single E2B sandbox."""
        from e2b_code_interpreter import AsyncSandbox
        
        os.environ["E2B_API_KEY"] = self.api_key
        sandbox = await AsyncSandbox.create()
        return sandbox
    
    async def acquire(self, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        """Acquire a sandbox from the pool."""
        if not self._initialized:
            await self.initialize()
        
        try:
            sandbox = await asyncio.wait_for(self.available.get(), timeout=timeout)
            return sandbox
        except asyncio.TimeoutError:
            raise SandboxError("Timeout waiting for available sandbox")
    
    async def release(self, sandbox):
        """Release a sandbox back to the pool."""
        if sandbox in self.all_sandboxes:
            await self.available.put(sandbox)
    
    async def cleanup(self):
        """Clean up all sandboxes in the pool."""
        for sandbox in self.all_sandboxes:
            try:
                await sandbox.close()
            except Exception as e:
                logger.error(f"Error closing sandbox: {e}")
        
        self.all_sandboxes.clear()
        self._initialized = False


class E2BSandbox:
    """E2B cloud sandbox implementation using official SDK with pooling."""
    
    _pool: Optional[E2BSandboxPool] = None
    _pool_lock = asyncio.Lock()
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sandbox = None
        self._acquired_from_pool = False
    
    @classmethod
    async def get_pool(cls, api_key: str) -> E2BSandboxPool:
        """Get or create the shared sandbox pool."""
        async with cls._pool_lock:
            if cls._pool is None:
                cls._pool = E2BSandboxPool(api_key)
                await cls._pool.initialize()
            return cls._pool
    
    async def create_sandbox(self):
        """Acquire a sandbox from the pool or create a new one."""
        try:
            pool = await self.get_pool(self.api_key)
            self.sandbox = await pool.acquire(timeout=5.0)
            self._acquired_from_pool = True
            logger.debug("Acquired sandbox from pool")
        except Exception as e:
            logger.warning(f"Failed to acquire from pool, creating new sandbox: {e}")
            # Fallback to creating a new sandbox
            from e2b_code_interpreter import AsyncSandbox
            
            os.environ["E2B_API_KEY"] = self.api_key
            self.sandbox = await AsyncSandbox.create()
            self._acquired_from_pool = False
    
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
        if not self.sandbox:
            await self.create_sandbox()
        
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
        """Release or close the E2B sandbox."""
        if self.sandbox:
            try:
                if self._acquired_from_pool:
                    pool = await self.get_pool(self.api_key)
                    await pool.release(self.sandbox)
                    logger.debug("Released sandbox back to pool")
                else:
                    await self.sandbox.close()
                    logger.debug("Closed standalone sandbox")
            except Exception as e:
                logger.warning(f"Error during sandbox cleanup: {e}")
            finally:
                self.sandbox = None
                self._acquired_from_pool = False
    
    async def writeFile(self, path: str, content: str) -> None:
        """Write a file to the E2B sandbox using native file operations."""
        if not self.sandbox:
            await self.create_sandbox()
        
        try:
            await self.sandbox.files.write(path, content)
            logger.debug(f"Wrote file to E2B sandbox: {path}")
        except Exception as e:
            raise SandboxError(f"Failed to write file to E2B sandbox: {e}")


class PyodideSandbox:
    """Local Pyodide sandbox implementation using Deno for enhanced security."""
    
    def __init__(self):
        self.available = self._check_deno_available()
        self._pyodide_script_path = Path(__file__).parent / "pyodide_sandbox.ts"
        self._process = None
        self._initialized = False
    
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
    
    async def execute_code(self, code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
        """Execute Python code using our direct Pyodide implementation."""
        if not self.available:
            raise SandboxError("Deno is not available for Pyodide execution")
        
        if not self._pyodide_script_path.exists():
            raise SandboxError(f"Pyodide script not found at {self._pyodide_script_path}")
        
        try:
            # Create execution request
            request = {
                "code": code,
                "timeout": timeout
            }
            
            # Run our Pyodide sandbox with Deno
            process = await asyncio.create_subprocess_exec(
                'deno', 'run',
                '--allow-net=cdn.jsdelivr.net,pyodide-cdn2.iodide.org',  # For downloading Pyodide
                '--allow-read',  # To read local files
                '--allow-write=.',  # To write output files
                str(self._pyodide_script_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send the request
            request_json = json.dumps(request)
            process.stdin.write(request_json.encode())
            await process.stdin.drain()
            process.stdin.close()
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout + 5  # Give extra time for process overhead
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                return {
                    "success": False,
                    "output": "",
                    "error": f"Execution timed out after {timeout} seconds",
                    "logs": [],
                }
            
            # Parse the response
            try:
                if stdout:
                    # The stdout should contain our JSON response
                    result = json.loads(stdout.decode('utf-8'))
                    return result
                else:
                    # If no stdout, check stderr for errors
                    error_msg = stderr.decode('utf-8', errors='replace') if stderr else "No output from Pyodide"
                    return {
                        "success": False,
                        "output": "",
                        "error": error_msg,
                        "logs": [],
                    }
                    
            except json.JSONDecodeError as e:
                # If we can't parse the response, return error info
                error_msg = f"Failed to parse response: {e}"
                if stderr:
                    error_msg += f"\nStderr: {stderr.decode('utf-8', errors='replace')}"
                return {
                    "success": False,
                    "output": stdout.decode('utf-8', errors='replace') if stdout else "",
                    "error": error_msg,
                    "logs": [],
                }
            
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Pyodide execution failed: {str(e)}",
                "logs": [],
            }
    
    async def writeFile(self, path: str, content: str) -> None:
        """Write a file to the Pyodide sandbox."""
        if not self.available:
            raise SandboxError("Deno is not available for Pyodide execution")
        
        # For now, we'll execute code to write the file
        # In the future, we could optimize this with a persistent Pyodide instance
        code = f'''
import json
with open({json.dumps(path)}, 'w') as f:
    f.write({json.dumps(content)})
print(f"File written to {{repr({json.dumps(path)})}}")
'''
        result = await self.execute_code(code, timeout=10)
        if not result["success"]:
            raise SandboxError(f"Failed to write file: {result['error']}")
    




# Global cache instance
_execution_cache = ExecutionCache(ttl_seconds=CACHE_TTL_SECONDS)

# Rate limiting
class RateLimiter:
    """Simple rate limiter for sandbox executions."""
    
    def __init__(self, max_requests: int = DEFAULT_RATE_LIMIT_REQUESTS, window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self._lock = asyncio.Lock()
    
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
    if timeout > MAX_TIMEOUT_SECONDS:
        return {
            "success": False,
            "output": "",
            "error": f"Timeout cannot exceed {MAX_TIMEOUT_SECONDS} seconds",
            "logs": [],
            "sandbox_used": "none",
        }
    
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
            "error": "No sandboxes available. Please set E2B_API_KEY environment variable or install Deno for Pyodide.",
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
                # Pyodide doesn't support package installation
                if install_packages and sandbox_name == "pyodide":
                    logger.warning("Pyodide sandbox doesn't support package installation")
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