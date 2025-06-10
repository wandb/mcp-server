"""
Fixed version of test_sandbox_execution.py that uses realistic testing approaches.
This demonstrates how to test sandbox functionality without excessive mocking.
"""

import asyncio
import os
import pytest
from unittest.mock import patch, Mock

from wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code import (
    execute_sandbox_code,
    check_sandbox_availability,
)
from wandb_mcp_server.mcp_tools.code_sandbox.pyodide_sandbox import PyodideSandbox
from wandb_mcp_server.mcp_tools.code_sandbox.e2b_sandbox import E2BSandbox
from wandb_mcp_server.mcp_tools.code_sandbox.test_helpers import (
    TestSandboxManager,
    IntegrationTestHelper,
)


class TestPyodideSandboxRealistic:
    """Realistic tests for Pyodide sandbox."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup and cleanup for each test."""
        self.manager = TestSandboxManager()
        await self.manager.__aenter__()
        yield
        await self.manager.__aexit__(None, None, None)
    
    def test_deno_availability_check(self):
        """Test Deno availability checking - this is a pure unit test."""
        # Test the actual check without mocking
        sandbox = PyodideSandbox()
        # This will be True if Deno is installed, False otherwise
        # Both are valid states to test
        assert isinstance(sandbox.available, bool)
        
        # If Deno is available, we can test further
        if sandbox.available:
            assert sandbox._pyodide_script_path.exists()
    
    @pytest.mark.asyncio
    async def test_pyodide_real_execution(self):
        """Test Pyodide execution with real Deno process if available."""
        sandbox = PyodideSandbox()
        
        if not sandbox.available:
            pytest.skip("Deno not available - this is OK")
        
        # Test real execution
        result = await sandbox.execute_code("print('Hello from Pyodide')")
        
        assert result["success"] is True
        assert "Hello from Pyodide" in result["output"]
        assert result["error"] is None or result["error"] == ""
    
    @pytest.mark.asyncio
    async def test_pyodide_error_handling(self):
        """Test real error handling in Pyodide."""
        sandbox = PyodideSandbox()
        
        if not sandbox.available:
            pytest.skip("Deno not available")
        
        # Test various error conditions
        error_cases = [
            ("1/0", "ZeroDivisionError"),
            ("undefined_var", "NameError"),
            ("import nonexistent", "ModuleNotFoundError"),
        ]
        
        for code, expected_error in error_cases:
            result = await sandbox.execute_code(code)
            assert result["success"] is False
            assert expected_error in result["error"] or expected_error in result["output"]
    
    @pytest.mark.asyncio
    async def test_pyodide_timeout_realistic(self):
        """Test timeout handling with realistic scenarios."""
        sandbox = PyodideSandbox()
        
        if not sandbox.available:
            pytest.skip("Deno not available")
        
        # Test code that takes time
        code = """
import time
print("Starting sleep")
time.sleep(5)
print("Finished sleep")
"""
        
        # Execute with 2 second timeout
        start = asyncio.get_event_loop().time()
        result = await sandbox.execute_code(code, timeout=2)
        elapsed = asyncio.get_event_loop().time() - start
        
        assert result["success"] is False
        assert "timeout" in result["error"].lower()
        # Should timeout within reasonable time (2s + overhead)
        assert elapsed < 4.0
    
    @pytest.mark.asyncio
    async def test_pyodide_process_persistence(self):
        """Test that Pyodide process is reused across executions."""
        sandbox = PyodideSandbox()
        
        if not sandbox.available:
            pytest.skip("Deno not available")
        
        # First execution
        result1 = await sandbox.execute_code("print('First execution')")
        assert result1["success"] is True
        
        # Get process reference
        process1 = PyodideSandbox._shared_process
        
        # Second execution
        result2 = await sandbox.execute_code("print('Second execution')")
        assert result2["success"] is True
        
        # Should reuse the same process
        process2 = PyodideSandbox._shared_process
        assert process1 is process2


class TestE2BSandboxRealistic:
    """Realistic tests for E2B sandbox."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup and cleanup for each test."""
        self.manager = TestSandboxManager()
        await self.manager.__aenter__()
        yield
        await self.manager.__aexit__(None, None, None)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set")
    async def test_e2b_real_execution(self):
        """Test real E2B execution."""
        sandbox = E2BSandbox(os.getenv("E2B_API_KEY"))
        
        async with sandbox:
            result = await sandbox.execute_code("print('Hello from E2B')")
            
            assert result["success"] is True
            assert "Hello from E2B" in result["output"]
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("E2B_API_KEY"), reason="E2B_API_KEY not set")
    async def test_e2b_package_installation(self):
        """Test real package installation in E2B."""
        sandbox = E2BSandbox(os.getenv("E2B_API_KEY"))
        
        async with sandbox:
            # Install a small, fast package
            success = await sandbox.install_packages(["requests"])
            assert success is True
            
            # Verify it's installed
            result = await sandbox.execute_code("""
import requests
print(f"Requests version: {requests.__version__}")
""")
            assert result["success"] is True
            assert "Requests version:" in result["output"]


class TestMainExecutionRealistic:
    """Realistic tests for the main execution function."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup and cleanup for each test."""
        self.manager = TestSandboxManager()
        await self.manager.__aenter__()
        yield
        await self.manager.__aexit__(None, None, None)
    
    @pytest.mark.asyncio
    async def test_auto_sandbox_selection(self):
        """Test that sandbox auto-selection works correctly."""
        available, types, _ = check_sandbox_availability()
        
        if not available:
            pytest.skip("No sandboxes available")
        
        # Don't specify sandbox type
        result = await execute_sandbox_code("print('auto selected')")
        
        assert result["success"] is True
        assert result["sandbox_used"] in types
        assert "auto selected" in result["output"]
    
    @pytest.mark.asyncio
    async def test_sandbox_type_specification(self):
        """Test specifying sandbox type works correctly."""
        available, types, _ = check_sandbox_availability()
        
        if "pyodide" in types:
            result = await execute_sandbox_code(
                "print('pyodide test')", 
                sandbox_type="pyodide"
            )
            assert result["success"] is True
            assert result["sandbox_used"] == "pyodide"
        
        if "e2b" in types:
            result = await execute_sandbox_code(
                "print('e2b test')", 
                sandbox_type="e2b"
            )
            assert result["success"] is True
            assert result["sandbox_used"] == "e2b"
    
    @pytest.mark.asyncio
    async def test_invalid_sandbox_type(self):
        """Test handling of invalid sandbox type."""
        result = await execute_sandbox_code(
            "print('test')", 
            sandbox_type="invalid_sandbox"
        )
        
        assert result["success"] is False
        assert "Invalid sandbox_type" in result["error"]
    
    @pytest.mark.asyncio
    async def test_concurrent_executions_realistic(self):
        """Test concurrent executions work correctly."""
        available, _, _ = check_sandbox_availability()
        
        if not available:
            pytest.skip("No sandboxes available")
        
        # Create concurrent tasks
        tasks = []
        for i in range(3):
            code = f"print('Concurrent task {i}')"
            tasks.append(execute_sandbox_code(code))
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for i, result in enumerate(results):
            assert result["success"] is True
            assert f"Concurrent task {i}" in result["output"]
    
    @pytest.mark.asyncio
    async def test_execution_metrics(self):
        """Test that execution metrics are included."""
        available, _, _ = check_sandbox_availability()
        
        if not available:
            pytest.skip("No sandboxes available")
        
        result = await execute_sandbox_code("print('metrics test')")
        
        assert result["success"] is True
        assert "execution_time_ms" in result
        assert isinstance(result["execution_time_ms"], int)
        assert result["execution_time_ms"] >= 0


class TestSandboxAvailability:
    """Test sandbox availability checking."""
    
    def test_check_availability_with_env_vars(self):
        """Test availability checking with different environment configurations."""
        # Test with sandbox disabled
        with patch.dict(os.environ, {"DISABLE_CODE_SANDBOX": "1"}):
            available, types, reason = check_sandbox_availability()
            assert available is False
            assert len(types) == 0
            assert "disabled" in reason
        
        # Test normal operation (actual availability depends on environment)
        available, types, reason = check_sandbox_availability()
        # Should return consistent results
        assert isinstance(available, bool)
        assert isinstance(types, list)
        assert isinstance(reason, str)
        
        if available:
            assert len(types) > 0
        else:
            assert len(types) == 0