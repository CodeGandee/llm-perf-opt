# Integration Tests

This directory contains integration tests that verify interactions between components and external services.

## Purpose

Integration tests ensure that different parts of the system work together correctly and that interactions with external services (databases, APIs, file systems, etc.) function as expected. They:
- **Test Integration Points**: Verify component interactions
- **Use Real Services**: May connect to actual databases, APIs, or file systems
- **Validate End-to-End Flows**: Test complete workflows
- **Require Setup**: May need external services running locally

## When to Write Integration Tests

Use integration tests when:
- Testing interactions between multiple components
- Verifying database operations (queries, transactions)
- Testing file system operations (reading/writing files)
- Validating API interactions
- Checking message queue processing
- Testing with real external services (S3, MinIO, etc.)

## Directory Structure

Organize by functional area or service:

```
tests/integration/
├── benchmarks/
│   ├── test_benchmark_pipeline.py
│   └── test_results_storage.py
├── optimization/
│   ├── test_model_optimization_flow.py
│   └── test_quantization_pipeline.py
└── storage/
    ├── test_s3_operations.py
    └── test_local_storage.py
```

## Naming Conventions

- **Files**: `test_*.py` or `*_test.py`
- **Functions**: `test_<integration_scenario>`
- **Markers**: Use `@pytest.mark.integration` to identify integration tests

## Example Integration Test

```python
"""Integration tests for benchmark results storage."""

import pytest
import os
from llm_perf_opt.benchmarks.storage import ResultsStore


@pytest.mark.integration
class TestResultsStore:
    """Integration tests for results storage."""

    @pytest.fixture
    def temp_storage_path(self, tmp_path):
        """Provide temporary storage path."""
        return tmp_path / "results"

    @pytest.fixture
    def results_store(self, temp_storage_path):
        """Create a results store instance."""
        return ResultsStore(str(temp_storage_path))

    def test_save_and_load_results(self, results_store):
        """Test saving and loading benchmark results."""
        # Arrange
        test_results = {
            "model": "test-model",
            "throughput": 150.5,
            "latency": 0.05
        }

        # Act
        results_store.save("test_benchmark", test_results)
        loaded_results = results_store.load("test_benchmark")

        # Assert
        assert loaded_results == test_results

    @pytest.mark.skipif(
        not os.getenv("S3_ENDPOINT"),
        reason="S3_ENDPOINT not configured"
    )
    def test_save_to_s3(self, results_store):
        """Test saving results to S3."""
        # This test only runs if S3 is configured
        test_results = {"model": "test", "score": 100}
        results_store.save_to_s3("test_key", test_results)
        # Verify results were saved
        assert results_store.exists_in_s3("test_key")
```

## Best Practices

### Service Availability
Always check if required services are available:

```python
import pytest
import requests

def is_service_available(url):
    """Check if a service is available."""
    try:
        response = requests.get(url, timeout=1)
        return response.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(
    not is_service_available("http://localhost:9000"),
    reason="MinIO not running on localhost:9000"
)
def test_minio_upload():
    """Test uploading to MinIO."""
    # Test implementation
    pass
```

### Environment Configuration
Use environment variables for service endpoints:

```python
import os
import pytest

@pytest.fixture
def s3_config():
    """Get S3 configuration from environment."""
    endpoint = os.getenv("S3_ENDPOINT")
    if not endpoint:
        pytest.skip("S3_ENDPOINT not configured")
    return {
        "endpoint": endpoint,
        "access_key": os.getenv("S3_ACCESS_KEY", "minioadmin"),
        "secret_key": os.getenv("S3_SECRET_KEY", "minioadmin")
    }


def test_s3_operations(s3_config):
    """Test S3 operations with configured endpoint."""
    # Use s3_config to connect and test
    pass
```

### Cleanup
Always clean up resources after tests:

```python
import pytest

@pytest.fixture
def test_database():
    """Create and cleanup test database."""
    # Setup
    db = create_test_database()
    db.initialize()

    yield db

    # Cleanup
    db.drop_all_tables()
    db.close()


def test_database_operations(test_database):
    """Test database operations."""
    # Test runs with clean database
    # Cleanup happens automatically after test
    pass
```

### Pytest Markers
Mark integration tests for selective execution:

```python
import pytest

# Mark single test
@pytest.mark.integration
def test_integration_scenario():
    """Test integration scenario."""
    pass

# Mark entire class
@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database."""

    def test_query(self):
        pass

    def test_transaction(self):
        pass
```

## Running Integration Tests

```bash
# Run all integration tests
pixi run pytest tests/integration/

# Run only tests marked as integration
pixi run pytest -m integration

# Skip integration tests
pixi run pytest -m "not integration"

# Run specific integration test file
pixi run pytest tests/integration/storage/test_s3_operations.py

# Run with verbose output
pixi run pytest tests/integration/ -v

# Run with environment variables
S3_ENDPOINT=http://localhost:9000 pixi run pytest tests/integration/

# Stop at first failure
pixi run pytest tests/integration/ -x
```

## Environment Setup

### Required Services
Document required services in test docstrings:

```python
"""
Integration tests for S3 storage.

Required Services:
    - MinIO or S3-compatible service on http://localhost:9000

Environment Variables:
    - S3_ENDPOINT: S3 service endpoint (default: http://localhost:9000)
    - S3_ACCESS_KEY: Access key (default: minioadmin)
    - S3_SECRET_KEY: Secret key (default: minioadmin)

Setup:
    docker run -p 9000:9000 minio/minio server /data
"""
```

### Docker Compose
Consider using docker-compose for test services:

```yaml
# docker-compose.test.yml
version: '3'
services:
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data
```

## Common Patterns

### Testing File Operations
```python
def test_file_processing(tmp_path):
    """Test processing files from disk."""
    # Create test file
    test_file = tmp_path / "input.txt"
    test_file.write_text("test data")

    # Process file
    result = process_file(str(test_file))

    # Verify output
    output_file = tmp_path / "output.txt"
    assert output_file.exists()
    assert output_file.read_text() == "processed: test data"
```

### Testing API Calls
```python
import responses

@responses.activate
def test_api_integration():
    """Test integration with external API."""
    # Mock API response
    responses.add(
        responses.GET,
        "https://api.example.com/data",
        json={"result": "success"},
        status=200
    )

    # Call function that uses API
    result = fetch_external_data()

    # Verify
    assert result["result"] == "success"
```

### Testing Database Operations
```python
def test_database_transaction(test_database):
    """Test database transaction."""
    # Start transaction
    with test_database.transaction():
        test_database.insert("test_table", {"key": "value"})
        test_database.insert("test_table", {"key": "value2"})

    # Verify data persisted
    results = test_database.query("SELECT * FROM test_table")
    assert len(results) == 2
```

## Continuous Integration

Integration tests in CI:
- May run less frequently than unit tests
- Can be configured to run on schedule
- Should fail gracefully if services unavailable
- Consider using service containers in CI

Example CI configuration snippet:
```yaml
# .github/workflows/integration-tests.yml
services:
  minio:
    image: minio/minio
    ports:
      - 9000:9000
    env:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin

steps:
  - name: Run integration tests
    env:
      S3_ENDPOINT: http://localhost:9000
    run: pixi run pytest tests/integration/
```

## Guidelines

1. **Check Service Availability**: Always verify services are running before tests
2. **Use Fixtures for Setup**: Create reusable fixtures for common resources
3. **Clean Up Resources**: Ensure cleanup happens even if tests fail
4. **Skip Gracefully**: Use `pytest.skip` when dependencies unavailable
5. **Use Environment Variables**: Configure endpoints via environment
6. **Mark Tests Clearly**: Use `@pytest.mark.integration`
7. **Document Requirements**: List required services and setup steps
8. **Isolate Tests**: Each test should be independent
9. **Test Realistic Scenarios**: Use real-world data and workflows
10. **Keep Tests Focused**: Test one integration point at a time

## References

- [Pytest Markers](https://docs.pytest.org/en/stable/how-to/mark.html)
- [Testing with Docker](https://docs.docker.com/compose/)
- [Integration Testing Best Practices](https://martinfowler.com/articles/practical-test-pyramid.html)
