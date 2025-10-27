# Unit Tests

This directory contains fast, deterministic unit tests that verify individual components in isolation.

## Purpose

Unit tests ensure that individual functions, classes, and modules work correctly without external dependencies. They are:
- **Fast**: Should execute in milliseconds
- **Isolated**: No external services, I/O, or network calls
- **Deterministic**: Same input always produces same output
- **Hermetic**: Self-contained with no side effects

## Directory Structure

Organize tests to mirror the source code structure:

```
tests/unit/
├── benchmarks/
│   ├── test_metrics.py
│   └── test_harness.py
├── optimization/
│   ├── test_quantization.py
│   └── test_kernel_opt.py
└── profiling/
    ├── test_memory.py
    └── test_performance.py
```

## Naming Conventions

- **Files**: `test_*.py` or `*_test.py`
- **Classes**: `Test<ComponentName>` (optional, for grouping)
- **Functions**: `test_<what_is_being_tested>`

## Example Unit Test

```python
"""Tests for performance metrics calculation."""

import pytest
from llm_perf_opt.benchmarks.metrics import calculate_throughput


def test_calculate_throughput_basic():
    """Test basic throughput calculation with valid inputs."""
    # Arrange
    tokens = 1000
    time_seconds = 2.0

    # Act
    throughput = calculate_throughput(tokens, time_seconds)

    # Assert
    assert throughput == 500.0


def test_calculate_throughput_zero_time_raises_error():
    """Test that zero time raises ValueError."""
    # Arrange
    tokens = 1000
    time_seconds = 0.0

    # Act & Assert
    with pytest.raises(ValueError, match="time must be positive"):
        calculate_throughput(tokens, time_seconds)


@pytest.mark.parametrize("tokens,time,expected", [
    (100, 1.0, 100.0),
    (200, 2.0, 100.0),
    (500, 5.0, 100.0),
])
def test_calculate_throughput_parametrized(tokens, time, expected):
    """Test throughput calculation with various inputs."""
    assert calculate_throughput(tokens, time) == expected
```

## Best Practices

### Isolation
- **Mock External Dependencies**: Use `unittest.mock` or `pytest-mock` for external calls
- **No I/O**: Avoid file operations, database calls, or network requests
- **No Time Dependencies**: Mock `time.time()`, `datetime.now()`, etc.

### Structure
- **Arrange-Act-Assert**: Clear three-phase structure
- **One Concept Per Test**: Each test should verify one behavior
- **Descriptive Names**: Test names should explain what they verify

### Fixtures
Use pytest fixtures for common setup:

```python
@pytest.fixture
def sample_model_config():
    """Provide a sample model configuration."""
    return {
        "model_name": "test-model",
        "batch_size": 8,
        "max_length": 512
    }


def test_model_initialization(sample_model_config):
    """Test that model initializes with config."""
    model = Model(sample_model_config)
    assert model.batch_size == 8
```

### Parametrization
Use `@pytest.mark.parametrize` for testing multiple inputs:

```python
@pytest.mark.parametrize("input_val,expected", [
    (0, 0),
    (1, 1),
    (10, 100),
    (-5, 25),
])
def test_square_function(input_val, expected):
    """Test square function with various inputs."""
    assert square(input_val) == expected
```

## Running Unit Tests

```bash
# Run all unit tests
pixi run pytest tests/unit/

# Run specific subdirectory
pixi run pytest tests/unit/benchmarks/

# Run specific test file
pixi run pytest tests/unit/benchmarks/test_metrics.py

# Run specific test function
pixi run pytest tests/unit/benchmarks/test_metrics.py::test_calculate_throughput_basic

# Run with verbose output
pixi run pytest tests/unit/ -v

# Run with coverage
pixi run pytest tests/unit/ --cov=llm_perf_opt --cov-report=term-missing

# Run tests matching a pattern
pixi run pytest tests/unit/ -k "throughput"

# Stop at first failure
pixi run pytest tests/unit/ -x
```

## Common Patterns

### Testing Exceptions
```python
def test_invalid_input_raises_error():
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError, match="expected error message"):
        function_that_should_fail("invalid")
```

### Testing with Mocks
```python
from unittest.mock import Mock, patch

def test_function_with_external_call():
    """Test function that calls external service."""
    with patch('module.external_service') as mock_service:
        mock_service.return_value = "mocked result"
        result = function_using_service()
        assert result == "processed: mocked result"
        mock_service.assert_called_once()
```

### Testing Class Behavior
```python
class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""

    def test_initialization(self):
        """Test tracker initializes with default values."""
        tracker = PerformanceTracker()
        assert tracker.total_tokens == 0

    def test_add_measurement(self):
        """Test adding measurement updates state."""
        tracker = PerformanceTracker()
        tracker.add_measurement(100, 1.0)
        assert tracker.total_tokens == 100
```

## Guidelines

1. **Keep Tests Fast**: Unit tests should complete in milliseconds
2. **No External Dependencies**: Mock everything outside your code
3. **Test Edge Cases**: Include boundary conditions and error cases
4. **Clear Assertions**: Use specific assertions with clear failure messages
5. **Avoid Test Interdependence**: Each test should be independent
6. **Follow Source Structure**: Mirror the organization of source code
7. **Document Complex Tests**: Add docstrings explaining non-obvious logic

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [Effective Python Testing](https://realpython.com/pytest-python-testing/)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)
