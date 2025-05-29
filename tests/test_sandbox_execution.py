"""
Tests for sandbox code execution functionality.
"""

import asyncio
import json
import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from dotenv import load_dotenv

from wandb_mcp_server.mcp_tools.code_sandbox.execute_sandbox_code import (
    execute_sandbox_code,
    E2BSandbox,
    PyodideSandbox,
    RestrictedPythonSandbox,
    SandboxError,
)
from wandb_mcp_server.mcp_tools.code_sandbox.sandbox_models import (
    SandboxExecutionRequest,
    SandboxExecutionResult,
    SandboxType,
)

load_dotenv()


class TestRestrictedPythonSandbox:
    """Test RestrictedPython sandbox functionality."""

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
    async def test_error_handling(self):
        """Test error handling in RestrictedPython sandbox."""
        sandbox = RestrictedPythonSandbox()
        if not sandbox.available:
            pytest.skip("RestrictedPython not available")
        
        code = """
# This should cause an error
x = undefined_variable
"""
        result = await sandbox.execute_code(code)
        
        assert result["success"] is False
        assert "NameError" in result["error"] or "undefined_variable" in result["error"]

    @pytest.mark.asyncio
    async def test_restricted_access(self):
        """Test that restricted operations are blocked."""
        sandbox = RestrictedPythonSandbox()
        if not sandbox.available:
            pytest.skip("RestrictedPython not available")
        
        # Try to access file system (should be restricted)
        code = """
import os
os.listdir('/')
"""
        result = await sandbox.execute_code(code)
        
        # Should either fail to import or fail to execute
        assert result["success"] is False


class TestPyodideSandbox:
    """Test Pyodide sandbox functionality."""

    @pytest.mark.asyncio
    async def test_node_not_available(self):
        """Test behavior when Node.js is not available."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock Node.js not being available
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.return_value = (b"", b"command not found")
            mock_subprocess.return_value = mock_process
            
            sandbox = PyodideSandbox()
            
            with pytest.raises(Exception):
                await sandbox.execute_code("print('test')")

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in Pyodide sandbox."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess, \
             patch('asyncio.wait_for') as mock_wait_for:
            
            # Mock timeout
            mock_wait_for.side_effect = asyncio.TimeoutError()
            mock_process = AsyncMock()
            mock_subprocess.return_value = mock_process
            
            sandbox = PyodideSandbox()
            
            with pytest.raises(SandboxError, match="timed out"):
                await sandbox.execute_code("print('test')", timeout=1)


class TestE2BSandbox:
    """Test E2B sandbox functionality."""

    @pytest.mark.asyncio
    async def test_sandbox_creation_failure(self):
        """Test handling of sandbox creation failure."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.return_value.raise_for_status.side_effect = Exception("API Error")
            
            sandbox = E2BSandbox("fake-api-key")
            
            with pytest.raises(Exception):
                await sandbox.create_sandbox()

    @pytest.mark.asyncio
    async def test_code_execution_success(self):
        """Test successful code execution in E2B sandbox."""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful sandbox creation
            create_response = MagicMock()
            create_response.json.return_value = {"sandboxId": "test-sandbox-123"}
            
            # Mock successful code execution
            execute_response = MagicMock()
            execute_response.json.return_value = {
                "success": True,
                "output": "Hello from E2B!",
                "error": None,
                "logs": []
            }
            
            mock_client_instance = mock_client.return_value.__aenter__.return_value
            mock_client_instance.post.side_effect = [create_response, execute_response]
            
            sandbox = E2BSandbox("fake-api-key")
            result = await sandbox.execute_code("print('Hello from E2B!')")
            
            assert result["success"] is True
            assert result["output"] == "Hello from E2B!"

    @pytest.mark.asyncio
    async def test_sandbox_cleanup(self):
        """Test sandbox cleanup functionality."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = mock_client.return_value.__aenter__.return_value
            
            sandbox = E2BSandbox("fake-api-key")
            sandbox.sandbox_id = "test-sandbox-123"
            
            await sandbox.close_sandbox()
            
            # Verify delete was called
            mock_client_instance.delete.assert_called_once()
            assert sandbox.sandbox_id is None


class TestExecuteSandboxCode:
    """Test the main execute_sandbox_code function."""

    @pytest.mark.asyncio
    async def test_e2b_priority_with_api_key(self):
        """Test that E2B is prioritized when API key is available."""
        with patch.dict(os.environ, {"E2B_API_KEY": "test-key"}), \
             patch('wandb_mcp_server.mcp_tools.execute_sandbox_code.E2BSandbox') as mock_e2b:
            
            mock_sandbox = AsyncMock()
            mock_sandbox.execute_code.return_value = {
                "success": True,
                "output": "E2B result",
                "error": None,
                "logs": []
            }
            mock_e2b.return_value = mock_sandbox
            
            result = await execute_sandbox_code("print('test')")
            
            assert result["success"] is True
            assert result["sandbox_used"] == "e2b"
            mock_sandbox.close_sandbox.assert_called_once()

    @pytest.mark.asyncio
    async def test_pyodide_fallback(self):
        """Test fallback to Pyodide when E2B is not available."""
        with patch.dict(os.environ, {}, clear=True), \
             patch('asyncio.create_subprocess_exec') as mock_subprocess:
            
            # Mock Node.js availability check
            node_check_process = AsyncMock()
            node_check_process.returncode = 0
            node_check_process.communicate.return_value = (b"v18.0.0\n", b"")
            
            # Mock Pyodide execution
            pyodide_process = AsyncMock()
            pyodide_process.returncode = 0
            pyodide_process.communicate.return_value = (
                json.dumps({
                    "success": True,
                    "output": "Pyodide result",
                    "error": None,
                    "logs": []
                }).encode(),
                b""
            )
            
            mock_subprocess.side_effect = [node_check_process, pyodide_process]
            
            with patch('tempfile.NamedTemporaryFile'), \
                 patch('os.unlink'):
                
                result = await execute_sandbox_code("print('test')")
                
                assert result["success"] is True
                assert result["sandbox_used"] == "pyodide"

    @pytest.mark.asyncio
    async def test_restricted_python_final_fallback(self):
        """Test fallback to RestrictedPython when other options fail."""
        with patch.dict(os.environ, {}, clear=True), \
             patch('asyncio.create_subprocess_exec') as mock_subprocess:
            
            # Mock Node.js not available
            node_process = AsyncMock()
            node_process.returncode = 1
            mock_subprocess.return_value = node_process
            
            with patch('wandb_mcp_server.mcp_tools.execute_sandbox_code.RestrictedPythonSandbox') as mock_restricted:
                mock_sandbox = AsyncMock()
                mock_sandbox.execute_code.return_value = {
                    "success": True,
                    "output": "RestrictedPython result",
                    "error": None,
                    "logs": []
                }
                mock_restricted.return_value = mock_sandbox
                
                result = await execute_sandbox_code("print('test')")
                
                assert result["success"] is True
                assert result["sandbox_used"] == "restricted"

    @pytest.mark.asyncio
    async def test_force_sandbox_type(self):
        """Test forcing a specific sandbox type."""
        with patch('wandb_mcp_server.mcp_tools.execute_sandbox_code.RestrictedPythonSandbox') as mock_restricted:
            mock_sandbox = AsyncMock()
            mock_sandbox.execute_code.return_value = {
                "success": True,
                "output": "Forced RestrictedPython",
                "error": None,
                "logs": []
            }
            mock_restricted.return_value = mock_sandbox
            
            result = await execute_sandbox_code("print('test')", sandbox_type="restricted")
            
            assert result["success"] is True
            assert result["sandbox_used"] == "restricted"

    @pytest.mark.asyncio
    async def test_all_sandboxes_fail(self):
        """Test behavior when all sandbox options fail."""
        with patch.dict(os.environ, {}, clear=True), \
             patch('asyncio.create_subprocess_exec') as mock_subprocess:
            
            # Mock Node.js not available
            node_process = AsyncMock()
            node_process.returncode = 1
            mock_subprocess.return_value = node_process
            
            with patch('wandb_mcp_server.mcp_tools.execute_sandbox_code.RestrictedPythonSandbox') as mock_restricted:
                mock_sandbox = AsyncMock()
                mock_sandbox.execute_code.side_effect = Exception("RestrictedPython failed")
                mock_restricted.return_value = mock_sandbox
                
                result = await execute_sandbox_code("print('test')")
                
                assert result["success"] is False
                assert "All sandbox options failed" in result["error"]
                assert result["sandbox_used"] == "none"


class TestPydanticModels:
    """Test Pydantic models for sandbox functionality."""

    def test_sandbox_execution_request_validation(self):
        """Test SandboxExecutionRequest validation."""
        # Valid request
        request = SandboxExecutionRequest(
            code="print('hello')",
            timeout=60,
            sandbox_type=SandboxType.E2B,
            install_packages=["numpy", "pandas"]
        )
        assert request.code == "print('hello')"
        assert request.timeout == 60
        assert request.sandbox_type == SandboxType.E2B
        assert request.install_packages == ["numpy", "pandas"]

    def test_sandbox_execution_request_defaults(self):
        """Test SandboxExecutionRequest defaults."""
        request = SandboxExecutionRequest(code="print('hello')")
        assert request.timeout == 30
        assert request.sandbox_type is None
        assert request.install_packages is None

    def test_sandbox_execution_request_timeout_validation(self):
        """Test timeout validation in SandboxExecutionRequest."""
        # Too small timeout
        with pytest.raises(ValueError):
            SandboxExecutionRequest(code="print('hello')", timeout=0)
        
        # Too large timeout
        with pytest.raises(ValueError):
            SandboxExecutionRequest(code="print('hello')", timeout=400)

    def test_sandbox_execution_result(self):
        """Test SandboxExecutionResult model."""
        result = SandboxExecutionResult(
            success=True,
            output="Hello, World!",
            error=None,
            logs=["Starting execution", "Execution complete"],
            sandbox_used="e2b",
            execution_time_ms=150
        )
        assert result.success is True
        assert result.output == "Hello, World!"
        assert result.error is None
        assert len(result.logs) == 2
        assert result.sandbox_used == "e2b"
        assert result.execution_time_ms == 150

    def test_sandbox_execution_result_defaults(self):
        """Test SandboxExecutionResult defaults."""
        result = SandboxExecutionResult(
            success=False,
            sandbox_used="none"
        )
        assert result.output == ""
        assert result.error is None
        assert result.logs == []
        assert result.execution_time_ms is None


@pytest.mark.integration
class TestSandboxIntegration:
    """Integration tests for sandbox functionality."""

    @pytest.mark.asyncio
    async def test_simple_math_calculation(self):
        """Test simple mathematical calculation."""
        code = """
import math
result = math.sqrt(16) + math.pi
print(f"Result: {result}")
"""
        result = await execute_sandbox_code(code)
        
        assert result["success"] is True
        assert "Result:" in result["output"]

    @pytest.mark.asyncio
    async def test_data_processing(self):
        """Test basic data processing."""
        code = """
data = [1, 2, 3, 4, 5]
squared = [x**2 for x in data]
total = sum(squared)
print(f"Original: {data}")
print(f"Squared: {squared}")
print(f"Sum of squares: {total}")
"""
        result = await execute_sandbox_code(code)
        
        assert result["success"] is True
        assert "Original:" in result["output"]
        assert "Squared:" in result["output"]
        assert "Sum of squares: 55" in result["output"]

    @pytest.mark.asyncio
    async def test_syntax_error_handling(self):
        """Test handling of syntax errors."""
        code = """
# Invalid syntax
if True
    print("This should fail")
"""
        result = await execute_sandbox_code(code)
        
        assert result["success"] is False
        assert "error" in result and result["error"] is not None

    @pytest.mark.asyncio
    async def test_timeout_behavior(self):
        """Test timeout behavior with a very short timeout."""
        code = """
import time
time.sleep(2)  # Sleep longer than timeout
print("This should not complete")
"""
        result = await execute_sandbox_code(code, timeout=1)
        
        # Result depends on sandbox type, but should either timeout or complete quickly
        # We mainly want to ensure no exceptions are raised
        assert isinstance(result, dict)
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])