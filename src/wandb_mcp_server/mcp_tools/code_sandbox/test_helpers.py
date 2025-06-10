"""
Test helpers for sandbox testing that provide better isolation without excessive mocking.
"""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from .pyodide_sandbox import PyodideSandbox
from .e2b_sandbox import E2BSandbox


class TestSandboxManager:
    """Manages sandbox instances for testing with proper isolation."""
    
    def __init__(self):
        self._original_pyodide_state = {}
        self._original_e2b_state = {}
        self._test_id = str(uuid.uuid4())[:8]
    
    async def __aenter__(self):
        """Save original state and prepare for testing."""
        # Save original class-level state
        self._original_pyodide_state = {
            '_shared_process': PyodideSandbox._shared_process,
            '_process_lock': PyodideSandbox._process_lock,
            '_initialized': PyodideSandbox._initialized,
            '_initialization_error': PyodideSandbox._initialization_error,
        }
        
        self._original_e2b_state = {
            '_shared_sandbox': E2BSandbox._shared_sandbox,
            '_sandbox_lock': E2BSandbox._sandbox_lock,
            '_api_key': E2BSandbox._api_key,
        }
        
        # Reset to clean state
        PyodideSandbox._shared_process = None
        PyodideSandbox._process_lock = None
        PyodideSandbox._initialized = False
        PyodideSandbox._initialization_error = None
        
        E2BSandbox._shared_sandbox = None
        E2BSandbox._sandbox_lock = None
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Restore original state and cleanup."""
        # Cleanup any processes we created
        await self.cleanup_test_processes()
        
        # Restore original state
        for key, value in self._original_pyodide_state.items():
            setattr(PyodideSandbox, key, value)
            
        for key, value in self._original_e2b_state.items():
            setattr(E2BSandbox, key, value)
    
    async def cleanup_test_processes(self):
        """Clean up any processes created during testing."""
        # Clean up Pyodide
        if PyodideSandbox._shared_process is not None:
            try:
                if hasattr(PyodideSandbox._shared_process, 'terminate'):
                    PyodideSandbox._shared_process.terminate()
                    await asyncio.wait_for(
                        PyodideSandbox._shared_process.wait(), 
                        timeout=5.0
                    )
            except Exception:
                pass
        
        # Clean up E2B
        if E2BSandbox._shared_sandbox is not None:
            try:
                await E2BSandbox._shared_sandbox.close()
            except Exception:
                pass


class IsolatedSandboxTest:
    """Base class for sandbox tests that need isolation."""
    
    @asynccontextmanager
    async def isolated_sandbox_test(self):
        """Context manager for isolated sandbox testing."""
        async with TestSandboxManager():
            yield
    
    def get_test_file_path(self, filename: str) -> str:
        """Get a unique test file path to avoid conflicts."""
        test_id = str(uuid.uuid4())[:8]
        return f"/tmp/test_{test_id}_{filename}"


class IntegrationTestHelper:
    """Helper for integration tests that need real sandbox behavior."""
    
    @staticmethod
    async def verify_sandbox_execution(
        code: str, 
        expected_output: str = None,
        expected_error: str = None,
        sandbox_type: Optional[str] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute code and verify output without mocking.
        Returns the full result for additional assertions.
        """
        from .execute_sandbox_code import execute_sandbox_code
        
        result = await execute_sandbox_code(
            code=code,
            timeout=timeout,
            sandbox_type=sandbox_type
        )
        
        if expected_output is not None:
            assert expected_output in result["output"], (
                f"Expected output '{expected_output}' not found in: {result['output']}"
            )
        
        if expected_error is not None:
            assert expected_error in result.get("error", ""), (
                f"Expected error '{expected_error}' not found in: {result.get('error', '')}"
            )
        else:
            assert result["success"], f"Execution failed: {result.get('error', '')}"
        
        return result
    
    @staticmethod
    async def verify_file_operations(
        sandbox_type: Optional[str] = None
    ) -> bool:
        """Test file operations work correctly in a sandbox."""
        test_id = str(uuid.uuid4())[:8]
        filename = f"/tmp/test_file_{test_id}.txt"
        content = f"Test content {test_id}"
        
        # Write file
        write_code = f"""
with open('{filename}', 'w') as f:
    f.write('{content}')
print(f"Wrote to {filename}")
"""
        
        # Read file
        read_code = f"""
try:
    with open('{filename}', 'r') as f:
        content = f.read()
    print(f"Read from {filename}: {{content}}")
except FileNotFoundError:
    print(f"File {filename} not found")
"""
        
        # Execute write
        write_result = await IntegrationTestHelper.verify_sandbox_execution(
            write_code, 
            expected_output=f"Wrote to {filename}",
            sandbox_type=sandbox_type
        )
        
        # Execute read
        read_result = await IntegrationTestHelper.verify_sandbox_execution(
            read_code,
            sandbox_type=sandbox_type
        )
        
        # Check if file persisted (E2B) or not (Pyodide)
        if sandbox_type == "e2b" or (sandbox_type is None and os.getenv("E2B_API_KEY")):
            # E2B should persist files
            assert content in read_result["output"], "E2B should persist files"
            return True
        else:
            # Pyodide may or may not persist depending on implementation
            return content in read_result["output"]


class SandboxTestFixtures:
    """Common test fixtures for sandbox testing."""
    
    @staticmethod
    def get_simple_computation_tests():
        """Get simple computation test cases."""
        return [
            ("print(2 + 2)", "4"),
            ("print('Hello, World!')", "Hello, World!"),
            ("import math; print(math.sqrt(16))", "4.0"),
            ("print(list(range(5)))", "[0, 1, 2, 3, 4]"),
        ]
    
    @staticmethod
    def get_error_handling_tests():
        """Get error handling test cases."""
        return [
            ("1/0", "ZeroDivisionError"),
            ("import nonexistent_module", "ModuleNotFoundError"),
            ("raise ValueError('test error')", "ValueError: test error"),
            ("undefined_variable", "NameError"),
        ]
    
    @staticmethod
    def get_timeout_tests():
        """Get timeout test cases."""
        return [
            ("import time; time.sleep(100)", 2),  # Should timeout after 2 seconds
            ("while True: pass", 2),  # Infinite loop
        ]
    
    @staticmethod
    def get_data_analysis_tests():
        """Get data analysis test cases."""
        return [
            ("""
import pandas as pd
data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
df = pd.DataFrame(data)
print(df.describe())
""", "count"),
            ("""
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Mean: {np.mean(arr)}")
""", "Mean: 3.0"),
        ]