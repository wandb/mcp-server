"""
Integration tests for sandbox code execution functionality.
These tests exercise real sandbox implementations to catch communication and process management issues.
"""

import asyncio
import os
import pytest
import time

from wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code import (
    execute_sandbox_code,
    check_sandbox_availability,
)
from wandb_mcp_server.mcp_tools.code_sandbox.pyodide_sandbox import PyodideSandbox
from wandb_mcp_server.mcp_tools.code_sandbox.e2b_sandbox import E2BSandbox


class TestSandboxAvailability:
    """Test sandbox availability detection."""

    def test_check_sandbox_availability(self):
        """Test that sandbox availability is correctly detected."""
        available, types, reason = check_sandbox_availability()

        # Should return valid data
        assert isinstance(available, bool)
        assert isinstance(types, list)
        assert isinstance(reason, str)

        # If available, should have at least one type
        if available:
            assert len(types) > 0
            assert any(t in ["e2b", "pyodide"] for t in types)

        print(f"Sandbox availability: {available}")
        print(f"Available types: {types}")
        print(f"Reason: {reason}")


@pytest.mark.integration
class TestPyodideIntegration:
    """Integration tests for Pyodide sandbox."""

    @pytest.fixture(autouse=True)
    async def setup_and_cleanup(self):
        """Setup and cleanup for each test."""
        # Clear any existing shared process before test
        await PyodideSandbox.cleanup_shared_process()
        yield
        # Cleanup after test
        await PyodideSandbox.cleanup_shared_process()

    def test_pyodide_availability(self):
        """Test Pyodide availability detection."""
        sandbox = PyodideSandbox()

        # Should properly detect Deno availability
        if sandbox.available:
            print("Pyodide sandbox is available (Deno found)")
        else:
            print("Pyodide sandbox not available (Deno not found)")
            pytest.skip("Deno not available for Pyodide testing")

    @pytest.mark.asyncio
    async def test_pyodide_basic_execution(self):
        """Test basic code execution with Pyodide."""
        sandbox = PyodideSandbox()

        if not sandbox.available:
            pytest.skip("Deno not available for Pyodide testing")

        # Test simple execution
        result = await sandbox.execute_code("print('Hello from Pyodide!')")

        assert result["success"] is True
        assert result["error"] is None
        assert isinstance(result["output"], str)
        assert isinstance(result["logs"], list)

    @pytest.mark.asyncio
    async def test_pyodide_math_operations(self):
        """Test mathematical operations in Pyodide."""
        sandbox = PyodideSandbox()

        if not sandbox.available:
            pytest.skip("Deno not available for Pyodide testing")

        # Test math operations
        result = await sandbox.execute_code("""
import math
result = math.sqrt(16)
print(f"Square root of 16 is {result}")
""")

        assert result["success"] is True
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_pyodide_process_persistence(self):
        """Test that Pyodide process persists across multiple executions."""
        sandbox = PyodideSandbox()

        if not sandbox.available:
            pytest.skip("Deno not available for Pyodide testing")

        # Execute multiple times to test process persistence
        for i in range(3):
            result = await sandbox.execute_code(f"print('Execution {i + 1}')")

            assert result["success"] is True, (
                f"Execution {i + 1} failed: {result.get('error')}"
            )
            assert result["error"] is None

    @pytest.mark.asyncio
    async def test_pyodide_variable_persistence(self):
        """Test that variables persist across executions in the same process."""
        sandbox = PyodideSandbox()

        if not sandbox.available:
            pytest.skip("Deno not available for Pyodide testing")

        # Set a variable
        result1 = await sandbox.execute_code("test_var = 42")
        assert result1["success"] is True

        # Use the variable in next execution
        result2 = await sandbox.execute_code("print(f'test_var = {test_var}')")
        assert result2["success"] is True
        assert result2["error"] is None

    @pytest.mark.asyncio
    async def test_pyodide_error_handling(self):
        """Test error handling in Pyodide execution."""
        sandbox = PyodideSandbox()

        if not sandbox.available:
            pytest.skip("Deno not available for Pyodide testing")

        # Test syntax error
        result = await sandbox.execute_code("print('unclosed string")

        assert result["success"] is False
        assert result["error"] is not None
        assert "SyntaxError" in result["error"] or "syntax" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_pyodide_timeout_handling(self):
        """Test timeout handling in Pyodide."""
        sandbox = PyodideSandbox()

        if not sandbox.available:
            pytest.skip("Deno not available for Pyodide testing")

        # Test with very short timeout
        start_time = time.time()
        result = await sandbox.execute_code("import time; time.sleep(2)", timeout=1)
        end_time = time.time()

        # Should timeout quickly
        assert end_time - start_time < 10  # Should not take too long
        assert result["success"] is False
        assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_pyodide_package_loading_output(self):
        """Test that package loading messages don't interfere with JSON responses."""
        sandbox = PyodideSandbox()

        if not sandbox.available:
            pytest.skip("Deno not available for Pyodide testing")

        # This should trigger package loading messages but still return valid JSON
        result = await sandbox.execute_code("""
import numpy as np
import pandas as pd
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Sum: {np.sum(arr)}")
""")

        assert result["success"] is True, f"Failed with error: {result.get('error')}"
        assert result["error"] is None
        assert isinstance(result["output"], str)
        assert isinstance(result["logs"], list)

    @pytest.mark.asyncio
    async def test_pyodide_process_recovery(self):
        """Test that Pyodide can recover from process failures."""
        sandbox = PyodideSandbox()

        if not sandbox.available:
            pytest.skip("Deno not available for Pyodide testing")

        # First execution should work
        result1 = await sandbox.execute_code("print('First execution')")
        assert result1["success"] is True

        # Simulate process death by clearing the shared process
        async with PyodideSandbox._process_lock:
            if PyodideSandbox._shared_process:
                PyodideSandbox._shared_process.terminate()
                await PyodideSandbox._shared_process.wait()
            PyodideSandbox._shared_process = None

        # Next execution should recover and work
        result2 = await sandbox.execute_code("print('Second execution after recovery')")
        assert result2["success"] is True, f"Recovery failed: {result2.get('error')}"

    @pytest.mark.asyncio
    async def test_pyodide_file_operations(self):
        """Test file write operations in Pyodide."""
        sandbox = PyodideSandbox()

        if not sandbox.available:
            pytest.skip("Deno not available for Pyodide testing")

        # Test file writing
        try:
            await sandbox.writeFile("/tmp/test.txt", "Hello, Pyodide!")

            # Test reading the file back
            result = await sandbox.execute_code("""
with open('/tmp/test.txt', 'r') as f:
    content = f.read()
print(f"File content: {content}")
""")

            assert result["success"] is True
            assert "Hello, Pyodide!" in result["output"]

        except Exception as e:
            pytest.fail(f"File operations failed: {e}")


@pytest.mark.integration
class TestE2BIntegration:
    """Integration tests for E2B sandbox."""

    @pytest.fixture(autouse=True)
    async def setup_and_cleanup(self):
        """Setup and cleanup for each test."""
        # Cleanup any existing shared sandbox
        await E2BSandbox.cleanup_shared_sandbox()
        yield
        # Cleanup after test
        await E2BSandbox.cleanup_shared_sandbox()

    def test_e2b_availability(self):
        """Test E2B availability detection."""
        api_key = os.getenv("E2B_API_KEY")

        if not api_key:
            print("E2B sandbox not available (E2B_API_KEY not set)")
            pytest.skip("E2B_API_KEY not set")
        else:
            print("E2B sandbox is available (API key found)")

    @pytest.mark.asyncio
    async def test_e2b_basic_execution(self):
        """Test basic code execution with E2B."""
        api_key = os.getenv("E2B_API_KEY")
        if not api_key:
            pytest.skip("E2B_API_KEY not set")

        sandbox = E2BSandbox(api_key)

        try:
            result = await sandbox.execute_code("print('Hello from E2B!')")

            assert result["success"] is True
            assert result["error"] is None
            assert isinstance(result["output"], str)
            assert isinstance(result["logs"], list)

        finally:
            await sandbox.close_sandbox()

    @pytest.mark.asyncio
    async def test_e2b_package_installation(self):
        """Test package installation in E2B."""
        api_key = os.getenv("E2B_API_KEY")
        if not api_key:
            pytest.skip("E2B_API_KEY not set")

        sandbox = E2BSandbox(api_key)

        try:
            # Test package installation
            result = await sandbox.execute_code(
                "import requests; print('requests imported successfully')",
                install_packages=["requests"],
            )

            assert result["success"] is True
            assert "requests imported successfully" in result["output"]

        finally:
            await sandbox.close_sandbox()


@pytest.mark.integration
class TestMainExecutionIntegration:
    """Integration tests for the main execute_sandbox_code function."""

    @pytest.mark.asyncio
    async def test_execute_sandbox_code_auto_selection(self):
        """Test that execute_sandbox_code automatically selects available sandbox."""
        result = await execute_sandbox_code(
            "print('Hello from auto-selected sandbox!')"
        )

        # Should succeed with some sandbox
        if result["success"]:
            assert result["sandbox_used"] in ["e2b", "pyodide"]
            assert isinstance(result["output"], str)
            assert isinstance(result["logs"], list)
            assert "execution_time_ms" in result
        else:
            # If no sandboxes available, should have clear error message
            assert "No sandboxes available" in result["error"]
            assert result["sandbox_used"] == "none"

    @pytest.mark.asyncio
    async def test_execute_sandbox_code_with_timeout(self):
        """Test timeout handling in main execution function."""
        start_time = time.time()
        result = await execute_sandbox_code("import time; time.sleep(5)", timeout=2)
        end_time = time.time()

        # Should timeout quickly
        assert end_time - start_time < 10

        if result["success"] is False and "timeout" in result["error"].lower():
            # Timeout worked correctly
            assert "execution_time_ms" in result
        elif result["success"] is False and "No sandboxes available" in result["error"]:
            # No sandboxes available, which is also acceptable
            pytest.skip("No sandboxes available for timeout testing")
        else:
            pytest.fail(f"Unexpected result: {result}")

    @pytest.mark.asyncio
    async def test_execute_sandbox_code_error_recovery(self):
        """Test that the main function can recover from sandbox errors."""
        # First execution should work (or fail gracefully)
        result1 = await execute_sandbox_code("print('First execution')")

        # Second execution should also work (testing recovery)
        result2 = await execute_sandbox_code("print('Second execution')")

        # At least one should succeed if sandboxes are available
        if result1["success"] or result2["success"]:
            # If any succeeded, both should have valid structure
            for result in [result1, result2]:
                assert "success" in result
                assert "output" in result
                assert "error" in result
                assert "logs" in result
                assert "sandbox_used" in result
                assert "execution_time_ms" in result

    @pytest.mark.asyncio
    async def test_execute_sandbox_code_concurrent_executions(self):
        """Test concurrent executions don't interfere with each other."""
        # Run multiple executions concurrently
        tasks = []
        for i in range(3):
            task = execute_sandbox_code(f"print('Concurrent execution {i + 1}')")
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that all completed without exceptions
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), (
                f"Task {i} raised exception: {result}"
            )
            assert isinstance(result, dict), f"Task {i} returned non-dict: {result}"
            assert "success" in result
            assert "sandbox_used" in result


@pytest.mark.integration
class TestCommunicationProtocol:
    """Test communication protocol edge cases that caused the 'Connection lost' bug."""

    @pytest.mark.asyncio
    async def test_mixed_output_handling(self):
        """Test handling of mixed JSON and non-JSON output."""
        sandbox = PyodideSandbox()

        if not sandbox.available:
            pytest.skip("Deno not available for communication protocol testing")

        # This code might trigger package loading messages mixed with actual output
        result = await sandbox.execute_code("""
# This might trigger package loading
import matplotlib.pyplot as plt
import numpy as np

# Generate some output
print("Starting computation...")
x = np.linspace(0, 10, 100)
y = np.sin(x)
print(f"Generated {len(x)} data points")
print("Computation complete!")
""")

        # Should successfully parse JSON response despite any package loading messages
        assert isinstance(result, dict)
        assert "success" in result
        assert "output" in result
        assert "error" in result
        assert "logs" in result

    @pytest.mark.asyncio
    async def test_large_output_handling(self):
        """Test handling of large output that might cause buffer issues."""
        sandbox = PyodideSandbox()

        if not sandbox.available:
            pytest.skip("Deno not available for large output testing")

        # Generate large output
        result = await sandbox.execute_code("""
# Generate large output
for i in range(100):
    print(f"Line {i}: " + "x" * 100)
""")

        assert isinstance(result, dict)
        assert "success" in result
        if result["success"]:
            assert len(result["output"]) > 1000  # Should have substantial output

    @pytest.mark.asyncio
    async def test_rapid_successive_executions(self):
        """Test rapid successive executions to stress test process communication."""
        sandbox = PyodideSandbox()

        if not sandbox.available:
            pytest.skip("Deno not available for rapid execution testing")

        # Execute many small tasks rapidly
        for i in range(10):
            result = await sandbox.execute_code(f"print('Rapid execution {i + 1}')")
            assert result["success"] is True, (
                f"Rapid execution {i + 1} failed: {result.get('error')}"
            )


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_sandbox_integration.py -v -m integration
    pytest.main([__file__, "-v", "-m", "integration"])
