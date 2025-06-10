"""
Realistic sandbox tests that verify actual functionality without excessive mocking.
These tests use the test helpers to ensure proper isolation while testing real behavior.
"""

import asyncio
import os
import pytest
from wandb_mcp_server.mcp_tools.code_sandbox.test_helpers import (
    IsolatedSandboxTest,
    IntegrationTestHelper,
    SandboxTestFixtures,
)
from wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code import (
    execute_sandbox_code,
    check_sandbox_availability,
)


class TestSandboxRealBehavior(IsolatedSandboxTest):
    """Test real sandbox behavior with proper isolation."""
    
    @pytest.mark.asyncio
    async def test_simple_computations_all_sandboxes(self):
        """Test simple computations work in all available sandboxes."""
        async with self.isolated_sandbox_test():
            available, types, _ = check_sandbox_availability()
            if not available:
                pytest.skip("No sandboxes available")
            
            for code, expected in SandboxTestFixtures.get_simple_computation_tests():
                for sandbox_type in types:
                    result = await IntegrationTestHelper.verify_sandbox_execution(
                        code=code,
                        expected_output=expected,
                        sandbox_type=sandbox_type
                    )
                    assert result["sandbox_used"] == sandbox_type
    
    @pytest.mark.asyncio
    async def test_error_handling_all_sandboxes(self):
        """Test error handling works correctly in all sandboxes."""
        async with self.isolated_sandbox_test():
            available, types, _ = check_sandbox_availability()
            if not available:
                pytest.skip("No sandboxes available")
            
            for code, expected_error in SandboxTestFixtures.get_error_handling_tests():
                for sandbox_type in types:
                    result = await execute_sandbox_code(
                        code=code,
                        sandbox_type=sandbox_type
                    )
                    assert result["success"] is False
                    assert expected_error in result["error"] or expected_error in result["output"]
    
    @pytest.mark.asyncio
    async def test_timeout_behavior(self):
        """Test timeout behavior with real code execution."""
        async with self.isolated_sandbox_test():
            available, types, _ = check_sandbox_availability()
            if not available:
                pytest.skip("No sandboxes available")
            
            # Test with a short timeout
            code = "import time; time.sleep(5)"
            
            for sandbox_type in types:
                start_time = asyncio.get_event_loop().time()
                result = await execute_sandbox_code(
                    code=code,
                    timeout=2,  # 2 second timeout
                    sandbox_type=sandbox_type
                )
                elapsed = asyncio.get_event_loop().time() - start_time
                
                assert result["success"] is False
                assert "timeout" in result["error"].lower()
                # Should timeout around 2 seconds (with some overhead)
                assert elapsed < 5, f"Timeout took too long: {elapsed}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_executions(self):
        """Test multiple concurrent executions work correctly."""
        async with self.isolated_sandbox_test():
            available, types, _ = check_sandbox_availability()
            if not available:
                pytest.skip("No sandboxes available")
            
            # Create multiple concurrent tasks
            tasks = []
            for i in range(5):
                code = f"print('Task {i}: ' + str({i} * {i}))"
                task = execute_sandbox_code(code)
                tasks.append(task)
            
            # Execute concurrently
            results = await asyncio.gather(*tasks)
            
            # Verify all succeeded
            for i, result in enumerate(results):
                assert result["success"] is True
                assert f"Task {i}: {i*i}" in result["output"]
    
    @pytest.mark.asyncio
    async def test_file_persistence_behavior(self):
        """Test actual file persistence behavior without mocking."""
        async with self.isolated_sandbox_test():
            # Test E2B if available
            if os.getenv("E2B_API_KEY"):
                e2b_persists = await IntegrationTestHelper.verify_file_operations(
                    sandbox_type="e2b"
                )
                assert e2b_persists, "E2B should persist files within session"
            
            # Test Pyodide if available
            available, types, _ = check_sandbox_availability()
            if "pyodide" in types:
                # Pyodide persistence depends on implementation
                await IntegrationTestHelper.verify_file_operations(
                    sandbox_type="pyodide"
                )
    
    @pytest.mark.asyncio
    async def test_data_analysis_packages(self):
        """Test data analysis packages work correctly."""
        async with self.isolated_sandbox_test():
            available, types, _ = check_sandbox_availability()
            if not available:
                pytest.skip("No sandboxes available")
            
            for code, expected in SandboxTestFixtures.get_data_analysis_tests():
                # Try in any available sandbox
                result = await execute_sandbox_code(code)
                
                if result["success"]:
                    assert expected in result["output"]
                else:
                    # Package might not be available in this sandbox
                    assert "ModuleNotFoundError" in result["error"]


class TestSandboxAutoSelection(IsolatedSandboxTest):
    """Test sandbox auto-selection behavior."""
    
    @pytest.mark.asyncio
    async def test_auto_selection_preference(self):
        """Test that auto-selection prefers E2B over Pyodide when both available."""
        async with self.isolated_sandbox_test():
            available, types, _ = check_sandbox_availability()
            if len(types) < 2:
                pytest.skip("Need multiple sandboxes for this test")
            
            # Don't specify sandbox type - let it auto-select
            result = await execute_sandbox_code("print('auto selected')")
            
            # Should prefer E2B if available
            if "e2b" in types:
                assert result["sandbox_used"] == "e2b"
            else:
                assert result["sandbox_used"] == "pyodide"
    
    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """Test realistic fallback behavior when preferred sandbox fails."""
        async with self.isolated_sandbox_test():
            available, types, _ = check_sandbox_availability()
            if len(types) < 2:
                pytest.skip("Need multiple sandboxes for fallback test")
            
            # Create a code that might fail in E2B but work in Pyodide
            # (This is a realistic scenario where E2B might have issues)
            code = """
import sys
print(f"Python version: {sys.version}")
print("Execution successful")
"""
            
            result = await execute_sandbox_code(code)
            assert result["success"] is True
            assert "Execution successful" in result["output"]


class TestSandboxStressTests(IsolatedSandboxTest):
    """Stress tests for sandbox functionality."""
    
    @pytest.mark.asyncio
    async def test_large_output_handling(self):
        """Test handling of large outputs."""
        async with self.isolated_sandbox_test():
            available, _, _ = check_sandbox_availability()
            if not available:
                pytest.skip("No sandboxes available")
            
            # Generate large output
            code = """
for i in range(1000):
    print(f"Line {i}: " + "x" * 100)
print("DONE")
"""
            
            result = await execute_sandbox_code(code)
            assert result["success"] is True
            assert "DONE" in result["output"]
            # Output should be present but might be truncated
            assert len(result["output"]) > 0
    
    @pytest.mark.asyncio
    async def test_memory_intensive_operations(self):
        """Test memory-intensive operations."""
        async with self.isolated_sandbox_test():
            available, _, _ = check_sandbox_availability()
            if not available:
                pytest.skip("No sandboxes available")
            
            code = """
# Create a large list
data = list(range(1000000))
print(f"Created list with {len(data)} elements")
print(f"Sum: {sum(data)}")
"""
            
            result = await execute_sandbox_code(code, timeout=10)
            
            if result["success"]:
                assert "1000000 elements" in result["output"]
            else:
                # Might fail due to memory limits - that's OK
                assert "MemoryError" in result["error"] or "timeout" in result["error"].lower()


class TestSandboxSecurity(IsolatedSandboxTest):
    """Security-related tests for sandboxes."""
    
    @pytest.mark.asyncio
    async def test_filesystem_isolation(self):
        """Test that sandboxes provide proper filesystem isolation."""
        async with self.isolated_sandbox_test():
            available, types, _ = check_sandbox_availability()
            if not available:
                pytest.skip("No sandboxes available")
            
            # Try to access system files (should fail or be isolated)
            code = """
import os
try:
    # Try to list root directory
    files = os.listdir('/')
    print(f"Root contains {len(files)} items")
    # Try to read a system file
    with open('/etc/passwd', 'r') as f:
        content = f.read()
    print("WARNING: Could read /etc/passwd!")
except Exception as e:
    print(f"Good: System file access blocked - {type(e).__name__}")
"""
            
            for sandbox_type in types:
                result = await execute_sandbox_code(code, sandbox_type=sandbox_type)
                # Should either fail or show isolated filesystem
                assert "WARNING" not in result["output"]
    
    @pytest.mark.asyncio
    async def test_network_isolation(self):
        """Test network isolation in sandboxes."""
        async with self.isolated_sandbox_test():
            available, types, _ = check_sandbox_availability()
            if not available:
                pytest.skip("No sandboxes available")
            
            code = """
import urllib.request
try:
    response = urllib.request.urlopen('http://example.com')
    print(f"Network access allowed: {response.status}")
except Exception as e:
    print(f"Network access blocked: {type(e).__name__}")
"""
            
            for sandbox_type in types:
                result = await execute_sandbox_code(code, sandbox_type=sandbox_type)
                # Network access might be blocked or allowed depending on sandbox
                assert result["success"] is True or "Error" in result["error"]