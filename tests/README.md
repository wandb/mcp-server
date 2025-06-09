# Sandbox Testing Strategy

This directory contains comprehensive tests for the sandbox code execution functionality. The tests are organized into two main categories to catch different types of issues.

## Test Categories

### 1. Unit Tests (`test_sandbox_execution.py`)
- **Purpose**: Test business logic and API contracts
- **Method**: Heavy use of mocking to isolate components
- **Speed**: Fast (< 1 second)
- **Requirements**: None (no external dependencies)
- **Coverage**: Logic, error handling, caching, rate limiting

**Pros:**
- Fast execution
- Reliable (no external dependencies)
- Good for testing edge cases and error conditions
- Excellent for CI/CD pipelines

**Cons:**
- Don't catch integration issues
- May miss real-world communication problems
- Can give false confidence if mocks don't match reality

### 2. Integration Tests (`test_sandbox_integration.py`)
- **Purpose**: Test real sandbox implementations and communication protocols
- **Method**: Exercise actual sandbox environments
- **Speed**: Slower (5-30 seconds per test)
- **Requirements**: Deno for Pyodide, E2B_API_KEY for E2B
- **Coverage**: Process management, communication protocols, real execution

**Pros:**
- Catch real-world issues like the "Connection lost" bug
- Test actual communication protocols
- Verify process persistence and recovery
- Test package loading and output handling

**Cons:**
- Slower execution
- Require external dependencies
- May be flaky due to network issues
- Not suitable for all CI environments

## Why Both Are Needed

The "Connection lost" bug that was fixed demonstrates why both test types are essential:

1. **Unit tests** validated the logic and API contracts but missed the bug because they mocked the subprocess communication
2. **Integration tests** would have caught the bug because they exercise the real Pyodide process and communication protocol

## Running Tests

### Quick Start
```bash
# Run only unit tests (default, fast)
python -m pytest

# Run integration tests (requires sandbox environments)
python -m pytest -m integration

# Run all tests
python -m pytest -m "unit or integration"
```

### Using the Test Runner
```bash
# Check test environment and sandbox availability
python run_tests.py check

# Run unit tests only
python run_tests.py unit

# Run integration tests only
python run_tests.py integration

# Run all tests
python run_tests.py all
```

### Test Markers

Tests are marked with pytest markers for easy filtering:

- `@pytest.mark.integration` - Integration tests requiring real sandboxes
- `@pytest.mark.unit` - Unit tests using mocking
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.e2b` - Tests requiring E2B_API_KEY
- `@pytest.mark.pyodide` - Tests requiring Deno

## Test Environment Setup

### For Unit Tests
No special setup required - they use mocking.

### For Integration Tests

#### Pyodide Tests
```bash
# Install Deno
curl -fsSL https://deno.land/install.sh | sh
export PATH="$HOME/.deno/bin:$PATH"
```

#### E2B Tests
```bash
# Get API key from https://e2b.dev
export E2B_API_KEY="your_api_key_here"
```

## Test Structure

### Integration Test Classes

1. **TestSandboxAvailability** - Test sandbox detection
2. **TestPyodideIntegration** - Comprehensive Pyodide testing
   - Basic execution
   - Process persistence
   - Error handling
   - Timeout handling
   - Package loading (the bug that was missed)
   - Process recovery
   - File operations
3. **TestE2BIntegration** - E2B sandbox testing
4. **TestMainExecutionIntegration** - End-to-end testing
5. **TestCommunicationProtocol** - Protocol edge cases

### Key Integration Tests

#### Communication Protocol Tests
These tests specifically target the issues that caused the "Connection lost" bug:

- **Mixed output handling** - Tests that package loading messages don't interfere with JSON responses
- **Large output handling** - Tests buffer management
- **Rapid successive executions** - Stress tests process communication

#### Process Management Tests
- **Process persistence** - Verifies the same process handles multiple executions
- **Process recovery** - Tests recovery from process failures
- **Variable persistence** - Ensures state is maintained across executions

## CI/CD Integration

### Recommended CI Strategy

```yaml
# Fast feedback loop
unit_tests:
  runs-on: ubuntu-latest
  steps:
    - run: python -m pytest -m "not integration"

# Comprehensive testing (optional, slower)
integration_tests:
  runs-on: ubuntu-latest
  steps:
    - name: Install Deno
      run: curl -fsSL https://deno.land/install.sh | sh
    - name: Run integration tests
      run: python -m pytest -m integration
      env:
        E2B_API_KEY: ${{ secrets.E2B_API_KEY }}
```

## Adding New Tests

### When to Add Unit Tests
- Testing new business logic
- Testing error conditions
- Testing API contracts
- Testing caching/rate limiting

### When to Add Integration Tests
- Adding new sandbox implementations
- Changing communication protocols
- Adding new process management features
- After fixing bugs that unit tests missed

### Test Naming Convention
- Unit tests: `test_<functionality>`
- Integration tests: `test_<sandbox>_<functionality>`
- Communication tests: `test_<protocol_aspect>`

## Debugging Failed Tests

### Unit Test Failures
- Check mock setup
- Verify test logic
- Check for API changes

### Integration Test Failures
- Check sandbox availability (`python run_tests.py check`)
- Verify environment setup (Deno, API keys)
- Check network connectivity
- Look for process management issues

## Future Improvements

1. **Performance benchmarking** - Add tests that measure execution time
2. **Memory usage testing** - Monitor memory leaks in long-running processes
3. **Concurrent execution testing** - More stress testing of concurrent operations
4. **Network failure simulation** - Test behavior under network issues
5. **Resource exhaustion testing** - Test behavior when resources are limited 