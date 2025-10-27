# Test Suite

This directory contains the comprehensive test suite for the LLM Performance Optimization project.

## Test Organization

Tests are organized by purpose and execution context:

```
tests/
├── unit/              # Fast, deterministic unit tests
├── integration/       # I/O, service, or multi-component tests
└── manual/            # Manually executed scripts (not CI-collected)
```

## Test Types

### Unit Tests (`unit/`)
- **Purpose**: Test individual components in isolation
- **Characteristics**: Fast, deterministic, no external dependencies
- **Discovery**: Automatically collected by pytest in CI
- **Naming**: `test_*.py` or `*_test.py`
- **Structure**: Mirrors source code organization under subdirectories

### Integration Tests (`integration/`)
- **Purpose**: Test interactions with external services or I/O operations
- **Characteristics**: May require external services (databases, S3/MinIO, filesystem)
- **Discovery**: Collected by CI if enabled; can be skipped via pytest markers
- **Naming**: `test_*.py` or `*_test.py`
- **Environment**: May require environment variables for service endpoints

### Manual Tests (`manual/`)
- **Purpose**: Heavy or environment-specific checks (e.g., Blender, GPU-specific)
- **Characteristics**: Not automatically run in CI, requires manual execution
- **Discovery**: NOT collected by CI (prefixed with `manual_`)
- **Naming**: `manual_*.py`
- **Execution**: Run directly as scripts when needed

## Running Tests

### Using Pixi (Recommended)

```bash
# Run all unit tests
pixi run pytest tests/unit/

# Run all integration tests
pixi run pytest tests/integration/

# Run specific test file
pixi run pytest tests/unit/test_example.py

# Run with coverage
pixi run pytest tests/unit/ --cov=llm_perf_opt --cov-report=html

# Run tests matching a pattern
pixi run pytest -k "test_performance"
```

### Manual Test Scripts

```bash
# Execute a manual test script directly
pixi run python tests/manual/manual_benchmark.py
```

## Test Guidelines

### Writing Unit Tests
- Keep tests fast and isolated
- Mock external dependencies
- Use fixtures for common setup
- Follow the Arrange-Act-Assert pattern
- Test one concept per test function

### Writing Integration Tests
- Check for required services before running
- Use pytest markers to allow skipping (e.g., `@pytest.mark.integration`)
- Clean up resources after tests
- Use environment variables for configuration
- Fail gracefully if dependencies are unavailable

### Writing Manual Tests
- Prefix files with `manual_` to prevent CI collection
- Include clear docstrings explaining purpose and requirements
- Add argument parsing for configuration options
- Log progress and results clearly
- Document any special setup requirements

## Best Practices

1. **Use Pixi**: Always prefer `pixi run` over system Python to ensure consistent environments
2. **Keep Unit Tests Hermetic**: No external dependencies or I/O operations
3. **Skip Gracefully**: Integration tests should skip when dependencies are unavailable
4. **Organize by Module**: Mirror the source code structure in test subdirectories
5. **Test Coverage**: Aim for high coverage but prioritize meaningful tests over percentage
6. **Clear Naming**: Test names should clearly describe what they test
7. **Documentation**: Add docstrings to complex test functions

## Configuration

Test configuration is managed through:
- `pyproject.toml`: Pytest and coverage settings
- `pixi.toml`: Test runner task definitions
- Environment variables: Service endpoints and credentials (integration tests)

## Continuous Integration

- Unit tests run on every commit
- Integration tests may run on schedule or when explicitly triggered
- Manual tests are never run automatically in CI
- Coverage reports are generated and tracked over time
