"""
Pyodide local sandbox implementation using Deno for enhanced security.
"""

import asyncio
import asyncio.subprocess
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from wandb_mcp_server.utils import get_rich_logger
from .sandbox_utils import (
    SandboxError,
    validate_timeout,
    DEFAULT_TIMEOUT_SECONDS,
)

logger = get_rich_logger(__name__)


class PyodideSandbox:
    """Local Pyodide sandbox implementation using Deno for enhanced security."""

    # Class-level persistent process
    _shared_process = None
    _process_lock = asyncio.Lock()
    _initialized = False
    _initialization_error = None
    _pre_download_complete = False

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
                ["deno", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                logger.warning(
                    "Deno is installed but returned an error. "
                    "Please ensure Deno is properly installed. "
                    "Visit https://deno.land/manual/getting_started/installation for installation instructions."
                )
                return False
            return True
        except FileNotFoundError:
            logger.warning(
                "Deno is not installed. Pyodide sandbox requires Deno for secure local execution. "
                "To install Deno:\n"
                "  - macOS/Linux: curl -fsSL https://deno.land/install.sh | sh\n"
                "  - Windows: irm https://deno.land/install.ps1 | iex\n"
                "  - Or visit: https://deno.land/manual/getting_started/installation"
            )
            return False
        except subprocess.TimeoutExpired:
            logger.warning(
                "Deno check timed out. Please ensure Deno is properly installed."
            )
            return False
        except Exception as e:
            logger.warning(f"Error checking for Deno: {e}")
            return False

    @classmethod
    def _is_process_alive(cls) -> bool:
        """Check if the shared process is alive and responsive."""
        if cls._shared_process is None:
            return False

        # Check if process has terminated
        if cls._shared_process.returncode is not None:
            return False

        # Check if stdin/stdout are still open
        try:
            if cls._shared_process.stdin.is_closing():
                return False
            if cls._shared_process.stdout.at_eof():
                return False
        except Exception:
            return False

        return True

    @classmethod
    async def get_or_create_process(cls, script_path: Path):
        """Get or create the shared Pyodide process."""
        async with cls._process_lock:
            # Check if we have a cached initialization error
            if cls._initialization_error:
                raise cls._initialization_error

            # Check if we need to create a new process
            if not cls._is_process_alive():
                # Clean up old process if it exists
                if cls._shared_process is not None:
                    try:
                        if cls._shared_process.returncode is None:
                            cls._shared_process.terminate()
                            try:
                                await asyncio.wait_for(
                                    cls._shared_process.wait(), timeout=2.0
                                )
                            except asyncio.TimeoutError:
                                cls._shared_process.kill()
                                await cls._shared_process.wait()
                    except Exception as e:
                        logger.debug(f"Error cleaning up old process: {e}")
                    finally:
                        cls._shared_process = None
                        cls._initialized = False

                logger.info("Starting persistent Pyodide sandbox process")
                try:
                    cls._shared_process = await asyncio.create_subprocess_exec(
                        "deno",
                        "run",
                        "--allow-net",  # For downloading Pyodide and packages
                        "--allow-read",  # To read local files
                        "--allow-write",  # To write output files and cache
                        "--allow-env",  # For environment variables
                        str(script_path),
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    cls._initialized = True

                    # Read initialization messages
                    try:
                        # Wait for "ready" message with timeout
                        ready_task = asyncio.create_task(cls._read_until_ready())
                        await asyncio.wait_for(ready_task, timeout=60)
                        logger.info("Pyodide sandbox process initialized successfully")
                        cls._initialization_error = None  # Clear any previous error
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for Pyodide to initialize")
                        if cls._shared_process:
                            cls._shared_process.kill()
                            await cls._shared_process.wait()
                        cls._shared_process = None
                        error = SandboxError(
                            "Failed to initialize Pyodide sandbox. "
                            "This may be due to network issues downloading Pyodide. "
                            "Please check your internet connection and try again."
                        )
                        cls._initialization_error = error
                        raise error
                except FileNotFoundError:
                    error = SandboxError(
                        "Deno executable not found. Please install Deno to use the Pyodide sandbox.\n"
                        "Installation instructions: https://deno.land/manual/getting_started/installation"
                    )
                    cls._initialization_error = error
                    raise error
                except Exception as e:
                    error = SandboxError(
                        f"Failed to start Pyodide sandbox process: {e}"
                    )
                    cls._initialization_error = error
                    raise error

            return cls._shared_process

    @classmethod
    async def _read_until_ready(cls):
        """Read stderr until we see the ready message."""
        if not cls._shared_process:
            return

        while True:
            try:
                line = await asyncio.wait_for(
                    cls._shared_process.stderr.readline(), timeout=30
                )
                if not line:
                    break
                line_str = line.decode("utf-8").strip()
                logger.debug(f"Pyodide init: {line_str}")
                if "Pyodide sandbox server ready" in line_str:
                    break
                elif "Initializing Pyodide..." in line_str:
                    logger.info(
                        "Downloading and initializing Pyodide (this may take a minute on first run)..."
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout reading initialization message from Pyodide process"
                )
                break
            except Exception as e:
                logger.warning(f"Error reading initialization message: {e}")
                break

    async def _send_request_and_get_response(
        self, request: Dict[str, Any], timeout: int
    ) -> Dict[str, Any]:
        """Send a request to the Pyodide process and get the response with proper error handling."""
        max_retries = 2

        for attempt in range(max_retries):
            try:
                # Get or create the persistent process
                process = await self.get_or_create_process(self._pyodide_script_path)

                # Verify process is still alive before sending request
                if not self._is_process_alive():
                    if attempt < max_retries - 1:
                        logger.warning(
                            "Process died before sending request, retrying..."
                        )
                        # Force recreation on next attempt
                        async with self._process_lock:
                            self._shared_process = None
                        continue
                    else:
                        raise SandboxError("Pyodide process is not responsive")

                # Send request as a single line JSON
                request_json = json.dumps(request) + "\n"
                process.stdin.write(request_json.encode())
                await process.stdin.drain()

                # Read response with timeout - may need to read multiple lines to get JSON
                response_line = None
                max_read_attempts = 10  # Prevent infinite loop
                read_attempts = 0

                try:
                    while read_attempts < max_read_attempts:
                        line = await asyncio.wait_for(
                            process.stdout.readline(),
                            timeout=timeout + 5,  # Give extra time for process overhead
                        )

                        if not line:
                            break

                        line_text = line.decode("utf-8").strip()

                        # Skip empty lines
                        if not line_text:
                            read_attempts += 1
                            continue

                        # Check if this looks like a JSON response (starts with { or [)
                        if line_text.startswith("{") or line_text.startswith("["):
                            response_line = line
                            break
                        else:
                            # This is likely a package loading message or other output - skip it
                            read_attempts += 1
                            continue

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout waiting for response from Pyodide after {timeout + 5} seconds"
                    )
                    # Don't kill the process on timeout, just return error
                    return {
                        "success": False,
                        "output": "",
                        "error": f"Execution timed out after {timeout} seconds",
                        "logs": [],
                    }

                # Parse the response
                if response_line:
                    try:
                        response_text = response_line.decode("utf-8").strip()
                        if not response_text:
                            logger.warning("Received empty response from Pyodide")
                            if attempt < max_retries - 1:
                                async with self._process_lock:
                                    self._shared_process = None
                                continue
                            else:
                                return {
                                    "success": False,
                                    "output": "",
                                    "error": "Received empty response from Pyodide",
                                    "logs": [],
                                }
                        result = json.loads(response_text)
                        return result
                    except json.JSONDecodeError as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Failed to parse response, retrying: {e}")
                            logger.warning(
                                f"Raw response was: {repr(response_line.decode('utf-8', errors='replace'))}"
                            )
                            # Force recreation on next attempt
                            async with self._process_lock:
                                self._shared_process = None
                            continue
                        else:
                            return {
                                "success": False,
                                "output": "",
                                "error": f"Failed to parse response: {e}",
                                "logs": [],
                            }
                else:
                    if attempt < max_retries - 1:
                        logger.warning("No response from Pyodide process, retrying...")
                        # Force recreation on next attempt
                        async with self._process_lock:
                            self._shared_process = None
                        continue
                    else:
                        return {
                            "success": False,
                            "output": "",
                            "error": "No response from Pyodide process",
                            "logs": [],
                        }

            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Communication error with Pyodide process, retrying: {e}"
                    )
                    # Force recreation on next attempt
                    async with self._process_lock:
                        self._shared_process = None
                    continue
                else:
                    return {
                        "success": False,
                        "output": "",
                        "error": f"Connection lost to Pyodide process: {str(e)}",
                        "logs": [],
                    }
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Unexpected error communicating with Pyodide process, retrying: {e}"
                    )
                    # Force recreation on next attempt
                    async with self._process_lock:
                        self._shared_process = None
                    continue
                else:
                    return {
                        "success": False,
                        "output": "",
                        "error": f"Pyodide execution failed: {str(e)}",
                        "logs": [],
                    }

        # Should never reach here, but just in case
        return {
            "success": False,
            "output": "",
            "error": "Maximum retries exceeded",
            "logs": [],
        }

    async def execute_code(
        self, code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS
    ) -> Dict[str, Any]:
        """Execute Python code using our persistent Pyodide process."""
        # Validate timeout
        timeout = validate_timeout(timeout)

        if not self.available:
            raise SandboxError(
                "Deno is not available for Pyodide execution.\n"
                "To install Deno:\n"
                "  - macOS/Linux: curl -fsSL https://deno.land/install.sh | sh\n"
                "  - Windows: irm https://deno.land/install.ps1 | iex\n"
                "  - Or visit: https://deno.land/manual/getting_started/installation"
            )

        if not self._pyodide_script_path.exists():
            raise SandboxError(
                f"Pyodide script not found at {self._pyodide_script_path}"
            )

        # Create execution request
        request = {"type": "execute", "code": code, "timeout": timeout}

        return await self._send_request_and_get_response(request, timeout)

    async def writeFile(self, path: str, content: str) -> None:
        """Write a file to the Pyodide sandbox using persistent process."""
        if not self.available:
            raise SandboxError(
                "Deno is not available for Pyodide execution. "
                "Install Deno with: curl -fsSL https://deno.land/install.sh | sh"
            )

        # Create file write request
        request = {"type": "writeFile", "path": path, "content": content}

        result = await self._send_request_and_get_response(request, 10)

        if not result.get("success", False):
            raise SandboxError(
                f"Failed to write file: {result.get('error', 'Unknown error')}"
            )

    @classmethod
    def pre_download_pyodide_sync(cls):
        """Pre-download Pyodide packages synchronously during startup."""
        if cls._pre_download_complete:
            return True

        sandbox = cls()
        if not sandbox.available:
            logger.info("Deno not available, skipping Pyodide pre-download")
            return False

        logger.info("Pre-downloading Pyodide packages...")
        try:
            # Run a temporary Deno process just to download Pyodide
            # This process will exit after downloading
            download_script = """
// Use the same npm package as the main script
import { loadPyodide } from "npm:pyodide@0.26.4";

console.error("Downloading Pyodide and packages...");
try {
    const pyodide = await loadPyodide();

    // Pre-load common packages
    await pyodide.loadPackage(["numpy", "pandas", "matplotlib"]);
    console.error("Pyodide packages downloaded successfully");
} catch (error) {
    console.error("Error downloading Pyodide:", error.message);
    Deno.exit(1);
}
Deno.exit(0);
"""

            # Create a temporary file for the download script
            import tempfile
            import subprocess

            with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
                f.write(download_script)
                temp_script_path = f.name

            try:
                # Run the download script synchronously
                result = subprocess.run(
                    ["deno", "run", "--allow-net", temp_script_path],
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout for download
                )

                if result.returncode == 0:
                    logger.info("Pyodide packages pre-downloaded successfully")
                    cls._pre_download_complete = True
                    return True
                else:
                    logger.warning(f"Pyodide pre-download failed: {result.stderr}")
                    return False

            finally:
                # Clean up temp file
                import os

                try:
                    os.unlink(temp_script_path)
                except Exception:
                    pass

        except subprocess.TimeoutExpired:
            logger.warning("Pyodide pre-download timed out")
            return False
        except Exception as e:
            logger.warning(f"Failed to pre-download Pyodide: {e}")
            return False

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
                        logger.warning(
                            "Pyodide process did not terminate gracefully, forcing kill"
                        )
                        cls._shared_process.kill()
                        await cls._shared_process.wait()
                        logger.info("Pyodide process killed forcefully")
                except Exception as e:
                    logger.error(f"Error closing shared Pyodide process: {e}")
                finally:
                    cls._shared_process = None
                    cls._initialized = False
                    cls._initialization_error = None
