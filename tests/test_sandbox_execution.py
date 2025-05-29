"""
Unit tests for sandbox code execution functionality.
These tests use mocking to test the sandbox logic without requiring actual sandboxes.
"""

import asyncio
import json
import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock, Mock
from dotenv import load_dotenv

from wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code import (
    execute_sandbox_code,
    E2BSandbox,
    PyodideSandbox,
    RestrictedPythonSandbox,
    SandboxError,
    SecurityValidator,
    ExecutionCache,
    RateLimiter,
    E2BSandboxPool,
)
from wandb_mcp_server.mcp_tools.code_sandbox.sandbox_models import (
    SandboxExecutionRequest,
    SandboxExecutionResult,
    SandboxType,
)

load_dotenv()


class TestSecurityValidator:
    """Test the security validator for RestrictedPython."""
    
    def test_validate_safe_code(self):
        """Test that safe code passes validation."""
        safe_codes = [
            "print('Hello, world!')",
            "x = 2 + 2",
            "import math\nprint(math.pi)",
            "for i in range(10):\n    print(i)",
        ]
        
        for code in safe_codes:
            is_safe, error = SecurityValidator.validate_code(code)
            assert is_safe is True
            assert error is None
    
    def test_validate_dangerous_code(self):
        """Test that dangerous code is detected."""
        dangerous_codes = [
            ("__import__('os')", "__import__"),
            ("eval('2+2')", "eval"),
            ("exec('print(1)')", "exec"),
            ("compile('x=1', 'test', 'exec')", "compile"),
            ("open('/etc/passwd')", "open"),
        ]
        
        for code, pattern in dangerous_codes:
            is_safe, error = SecurityValidator.validate_code(code)
            assert is_safe is False
            assert pattern in error
    
    @patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.logger')
    def test_suspicious_imports_warning(self, mock_logger):
        """Test that suspicious imports generate warnings."""
        code = "import os\nimport subprocess\nprint('test')"
        
        is_safe, error = SecurityValidator.validate_code(code)
        
        # Should still be safe (just a warning)
        assert is_safe is True
        assert error is None
        
        # But should log a warning
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "os" in warning_msg
        assert "subprocess" in warning_msg


class TestExecutionCache:
    """Test the execution cache functionality."""
    
    def test_cache_basic_operations(self):
        """Test basic cache get/set operations."""
        cache = ExecutionCache(max_size=2)
        
        result1 = {"success": True, "output": "test1"}
        result2 = {"success": True, "output": "test2"}
        
        # Test cache miss
        assert cache.get("code1", "e2b") is None
        
        # Test cache set and hit
        cache.set("code1", "e2b", result1)
        cached = cache.get("code1", "e2b")
        assert cached == result1
        
        # Test different code
        cache.set("code2", "e2b", result2)
        assert cache.get("code2", "e2b") == result2
        
        # Test cache with packages
        cache.set("code1", "e2b", result1, packages=["numpy"])
        assert cache.get("code1", "e2b", packages=["numpy"]) == result1
        assert cache.get("code1", "e2b", packages=["pandas"]) is None  # Different packages
    
    def test_cache_eviction(self):
        """Test LRU cache eviction."""
        cache = ExecutionCache(max_size=2)
        
        cache.set("code1", "e2b", {"output": "1"})
        cache.set("code2", "e2b", {"output": "2"})
        cache.set("code3", "e2b", {"output": "3"})  # Should evict code1
        
        assert cache.get("code1", "e2b") is None  # Evicted
        assert cache.get("code2", "e2b") is not None
        assert cache.get("code3", "e2b") is not None
    
    @patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.datetime')
    def test_cache_expiration(self, mock_datetime):
        """Test cache TTL expiration."""
        from datetime import datetime, timedelta
        
        cache = ExecutionCache(ttl_seconds=300)  # 5 minutes
        
        # Set initial time
        initial_time = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = initial_time
        
        cache.set("code1", "e2b", {"output": "test"})
        
        # Test before expiration
        mock_datetime.now.return_value = initial_time + timedelta(minutes=4)
        assert cache.get("code1", "e2b") is not None
        
        # Test after expiration
        mock_datetime.now.return_value = initial_time + timedelta(minutes=6)
        assert cache.get("code1", "e2b") is None


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_basic(self):
        """Test basic rate limiting."""
        limiter = RateLimiter(max_requests=3, window_seconds=1)
        
        # First 3 requests should pass
        for _ in range(3):
            assert await limiter.check_rate_limit() is True
        
        # 4th request should fail
        assert await limiter.check_rate_limit() is False
    
    @pytest.mark.asyncio
    @patch('time.time')
    async def test_rate_limit_window(self, mock_time):
        """Test rate limit window expiration."""
        limiter = RateLimiter(max_requests=2, window_seconds=10)
        
        # Initial time
        mock_time.return_value = 1000
        
        # Use up the limit
        assert await limiter.check_rate_limit() is True
        assert await limiter.check_rate_limit() is True
        assert await limiter.check_rate_limit() is False
        
        # Move time forward past the window
        mock_time.return_value = 1011  # 11 seconds later
        
        # Should be able to make requests again
        assert await limiter.check_rate_limit() is True


class TestE2BSandboxPool:
    """Test E2B sandbox pooling."""
    
    @pytest.mark.asyncio
    @patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.AsyncSandbox')
    async def test_pool_initialization(self, mock_sandbox_class):
        """Test pool initialization."""
        # Mock sandbox creation
        mock_sandbox = AsyncMock()
        mock_sandbox_class.create = AsyncMock(return_value=mock_sandbox)
        
        pool = E2BSandboxPool("test_api_key", pool_size=2)
        await pool.initialize()
        
        # Should create 2 sandboxes
        assert mock_sandbox_class.create.call_count == 2
        assert len(pool.all_sandboxes) == 2
    
    @pytest.mark.asyncio
    @patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.AsyncSandbox')
    async def test_pool_acquire_release(self, mock_sandbox_class):
        """Test acquiring and releasing sandboxes from pool."""
        mock_sandbox1 = AsyncMock()
        mock_sandbox2 = AsyncMock()
        mock_sandbox_class.create = AsyncMock(side_effect=[mock_sandbox1, mock_sandbox2])
        
        pool = E2BSandboxPool("test_api_key", pool_size=2)
        await pool.initialize()
        
        # Acquire sandboxes
        sb1 = await pool.acquire()
        sb2 = await pool.acquire()
        
        assert sb1 == mock_sandbox1
        assert sb2 == mock_sandbox2
        
        # Pool should be empty now
        with pytest.raises(asyncio.TimeoutError):
            await pool.acquire(timeout=0.1)
        
        # Release one back
        await pool.release(sb1)
        
        # Should be able to acquire again
        sb3 = await pool.acquire()
        assert sb3 == sb1  # Same sandbox


class TestE2BSandbox:
    """Test E2B sandbox implementation."""
    
    @pytest.mark.asyncio
    @patch('os.getenv')
    @patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.E2BSandboxPool')
    async def test_sandbox_pool_usage(self, mock_pool_class, mock_getenv):
        """Test that E2B uses the pool when available."""
        mock_getenv.return_value = "test_api_key"
        
        # Mock pool
        mock_pool = AsyncMock()
        mock_sandbox = AsyncMock()
        mock_pool.acquire = AsyncMock(return_value=mock_sandbox)
        mock_pool_class.return_value = mock_pool
        
        sandbox = E2BSandbox("test_api_key")
        await sandbox.create_sandbox()
        
        assert sandbox.sandbox == mock_sandbox
        assert sandbox._acquired_from_pool is True
    
    @pytest.mark.asyncio
    @patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.AsyncSandbox')
    async def test_package_installation(self, mock_sandbox_class):
        """Test package installation in E2B."""
        mock_sandbox = AsyncMock()
        mock_result = Mock()
        mock_result.exit_code = 0
        mock_result.stderr = ""
        mock_sandbox.commands.run = AsyncMock(return_value=mock_result)
        
        sandbox = E2BSandbox("test_api_key")
        sandbox.sandbox = mock_sandbox
        
        success = await sandbox.install_packages(["numpy", "pandas"])
        
        assert success is True
        mock_sandbox.commands.run.assert_called_once()
        call_args = mock_sandbox.commands.run.call_args[0][0]
        assert "pip install" in call_args
        assert "numpy" in call_args
        assert "pandas" in call_args
    
    @pytest.mark.asyncio
    @patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.AsyncSandbox')
    async def test_code_execution_via_file(self, mock_sandbox_class):
        """Test that code is executed via file write to avoid escaping issues."""
        mock_sandbox = AsyncMock()
        mock_result = Mock()
        mock_result.exit_code = 0
        mock_result.stdout = "Hello, world!"
        mock_result.stderr = ""
        mock_sandbox.commands.run = AsyncMock(return_value=mock_result)
        mock_sandbox.files.write = AsyncMock()
        
        sandbox = E2BSandbox("test_api_key")
        sandbox.sandbox = mock_sandbox
        
        code = "print('Hello, world!')"
        result = await sandbox.execute_code(code)
        
        # Should write code to file
        mock_sandbox.files.write.assert_called_once_with("/tmp/code_to_execute.py", code)
        
        # Should execute the file
        run_calls = mock_sandbox.commands.run.call_args_list
        assert len(run_calls) >= 1
        assert "python /tmp/code_to_execute.py" in run_calls[0][0][0]
        
        assert result["success"] is True
        assert result["output"] == "Hello, world!"


class TestPyodideSandbox:
    """Test Pyodide sandbox implementation."""
    
    @patch('subprocess.run')
    def test_nodejs_availability_check(self, mock_run):
        """Test Node.js availability checking."""
        # Test when Node.js is available
        mock_run.return_value = Mock(returncode=0)
        sandbox = PyodideSandbox()
        assert sandbox.available is True
        
        # Test when Node.js is not available
        mock_run.return_value = Mock(returncode=1)
        sandbox = PyodideSandbox()
        assert sandbox.available is False
        
        # Test when subprocess fails
        mock_run.side_effect = Exception("Command not found")
        sandbox = PyodideSandbox()
        assert sandbox.available is False
    
    @pytest.mark.asyncio
    @patch('os.path.exists')
    @patch('asyncio.create_subprocess_exec')
    async def test_pyodide_runner_usage(self, mock_subprocess, mock_exists):
        """Test that Pyodide uses the runner script when available."""
        mock_exists.return_value = True
        
        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(
            b'{"success": true, "output": "test output", "error": null}',
            b''
        ))
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        sandbox = PyodideSandbox()
        sandbox.available = True
        
        result = await sandbox.execute_code("print('test')")
        
        # Should call node with the runner script
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0]
        assert call_args[0] == 'node'
        assert 'pyodide_runner.js' in call_args[1]
        
        assert result["success"] is True
        assert result["output"] == "test output"


class TestRestrictedPythonSandbox:
    """Test RestrictedPython sandbox implementation."""
    
    @pytest.mark.asyncio
    async def test_simple_execution(self):
        """Test basic code execution in RestrictedPython sandbox."""
        sandbox = RestrictedPythonSandbox()
        if not sandbox.available:
            pytest.skip("RestrictedPython not available")
        
        code = """
print("Hello, World!")
result = 2 + 2
print(f"2 + 2 = {result}")
"""
        result = await sandbox.execute_code(code)
        
        assert result["success"] is True
        assert "Hello, World!" in result["output"]
        assert "2 + 2 = 4" in result["output"]
        assert result["error"] is None
    
    @pytest.mark.asyncio
    async def test_restricted_builtins(self):
        """Test that RestrictedPython has limited builtins."""
        sandbox = RestrictedPythonSandbox()
        if not sandbox.available:
            pytest.skip("RestrictedPython not available")
        
        # Test allowed builtins
        code = """
# These should work
print(len([1, 2, 3]))
print(sum([1, 2, 3]))
print(max([1, 2, 3]))
"""
        result = await sandbox.execute_code(code)
        assert result["success"] is True
        assert "3" in result["output"]
        assert "6" in result["output"]
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in RestrictedPython."""
        sandbox = RestrictedPythonSandbox()
        if not sandbox.available:
            pytest.skip("RestrictedPython not available")
        
        code = """
# Infinite loop
while True:
    pass
"""
        result = await sandbox.execute_code(code, timeout=1)
        
        assert result["success"] is False
        assert "timeout" in result["error"].lower()


class TestMainExecutionFunction:
    """Test the main execute_sandbox_code function."""
    
    @pytest.mark.asyncio
    @patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code._rate_limiter')
    async def test_rate_limiting_enforcement(self, mock_limiter):
        """Test that rate limiting is enforced."""
        mock_limiter.check_rate_limit = AsyncMock(return_value=False)
        
        result = await execute_sandbox_code("print('test')")
        
        assert result["success"] is False
        assert "Rate limit exceeded" in result["error"]
        assert result["sandbox_used"] == "none"
    
    @pytest.mark.asyncio
    @patch('os.getenv')
    async def test_security_validation_only_for_restricted(self, mock_getenv):
        """Test that security validation only applies to RestrictedPython."""
        mock_getenv.return_value = None  # No E2B API key
        
        # Mock PyodideSandbox to be unavailable
        with patch.object(PyodideSandbox, '_check_nodejs_available', return_value=False):
            # This should trigger security validation for RestrictedPython
            dangerous_code = "__import__('os').system('ls')"
            
            result = await execute_sandbox_code(dangerous_code, sandbox_type="restricted")
            
            assert result["success"] is False
            assert "Security validation failed" in result["error"]
            
            # But if we had E2B, it should not validate
            mock_getenv.return_value = "test_api_key"
            
            with patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.E2BSandbox') as mock_e2b:
                mock_sandbox = mock_e2b.return_value
                mock_sandbox.execute_code = AsyncMock(return_value={
                    "success": True,
                    "output": "executed",
                    "error": None,
                    "logs": []
                })
                mock_sandbox.close_sandbox = AsyncMock()
                
                result = await execute_sandbox_code(dangerous_code, sandbox_type="e2b")
                
                # Should execute without security validation
                assert "Security validation failed" not in result.get("error", "")
    
    @pytest.mark.asyncio
    @patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code._execution_cache')
    async def test_caching_behavior(self, mock_cache):
        """Test that caching works correctly."""
        # First call - cache miss
        mock_cache.get.return_value = None
        
        with patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.RestrictedPythonSandbox') as mock_sandbox:
            sandbox_instance = mock_sandbox.return_value
            sandbox_instance.available = True
            sandbox_instance.execute_code = AsyncMock(return_value={
                "success": True,
                "output": "test output",
                "error": None,
                "logs": []
            })
            
            code = "print('test')"
            result = await execute_sandbox_code(code)
            
            # Should check cache
            mock_cache.get.assert_called()
            
            # Should cache the result
            mock_cache.set.assert_called_once()
            
            # Second call - cache hit
            mock_cache.get.return_value = {
                "success": True,
                "output": "cached output",
                "error": None,
                "logs": [],
                "sandbox_used": "restricted"
            }
            
            result2 = await execute_sandbox_code(code)
            
            assert result2["output"] == "cached output"
            assert result2["execution_time_ms"] == 0  # Cached result
    
    @pytest.mark.asyncio
    async def test_fallback_behavior(self):
        """Test sandbox fallback when preferred options fail."""
        with patch('os.getenv', return_value="test_api_key"):
            with patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.E2BSandbox') as mock_e2b:
                with patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.PyodideSandbox') as mock_pyodide:
                    with patch('wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code.RestrictedPythonSandbox') as mock_restricted:
                        # E2B fails
                        e2b_instance = mock_e2b.return_value
                        e2b_instance.execute_code = AsyncMock(side_effect=Exception("E2B failed"))
                        e2b_instance.close_sandbox = AsyncMock()
                        
                        # Pyodide succeeds
                        pyodide_instance = mock_pyodide.return_value
                        pyodide_instance.available = True
                        pyodide_instance.execute_code = AsyncMock(return_value={
                            "success": True,
                            "output": "pyodide output",
                            "error": None,
                            "logs": []
                        })
                        
                        result = await execute_sandbox_code("print('test')")
                        
                        assert result["success"] is True
                        assert result["sandbox_used"] == "pyodide"
                        assert result["output"] == "pyodide output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])