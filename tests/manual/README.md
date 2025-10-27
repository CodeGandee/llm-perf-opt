# Manual Tests

This directory contains manually executed test scripts that are not run automatically in CI.

## Purpose

Manual tests are for scenarios that:
- **Require Heavy Resources**: GPU-intensive operations, large model loading
- **Need Specific Hardware**: Specialized accelerators, specific GPU models
- **Take Long Time**: Extended benchmark runs, stress tests
- **Require Manual Setup**: Complex environment configurations
- **Need Human Verification**: Visual inspection, interactive testing
- **Are Environment-Specific**: Platform-specific checks (Blender, specific drivers)

## When to Write Manual Tests

Create manual tests for:
- Performance benchmarks that take minutes/hours to run
- Tests requiring specific hardware (GPUs, TPUs, custom accelerators)
- Visual or interactive verification scenarios
- Stress tests with resource-intensive operations
- Platform-specific functionality
- Tests with complex external dependencies

## Naming Convention

**IMPORTANT**: All files must be prefixed with `manual_` to prevent pytest collection:

```
tests/manual/
├── manual_gpu_benchmark.py
├── manual_model_quantization.py
├── manual_stress_test.py
└── manual_visual_check.py
```

**DO NOT** name files `test_*.py` or `*_test.py` - they will be collected by pytest!

## Example Manual Test Script

```python
#!/usr/bin/env python3
"""
Manual benchmark for LLM inference performance on GPUs.

This script performs comprehensive GPU benchmarking and is not run in CI
due to hardware requirements and long execution time.

Requirements:
    - NVIDIA GPU with CUDA support
    - At least 16GB VRAM
    - PyTorch with CUDA enabled

Usage:
    pixi run python tests/manual/manual_gpu_benchmark.py --model gpt2
    pixi run python tests/manual/manual_gpu_benchmark.py --model gpt2 --iterations 100

Environment:
    CUDA_VISIBLE_DEVICES: Specify which GPUs to use (default: 0)

Expected Duration:
    ~30 minutes for default configuration
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm_perf_opt.benchmarks import GPUBenchmark
from llm_perf_opt.utils import check_cuda_available


def setup_logging():
    """Configure logging for the manual test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('manual_gpu_benchmark.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run manual GPU performance benchmark'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        help='Model name to benchmark (default: gpt2)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=50,
        help='Number of benchmark iterations (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for inference (default: 8)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('tmp/benchmark_results.json'),
        help='Output file for results (default: tmp/benchmark_results.json)'
    )
    return parser.parse_args()


def check_requirements(logger):
    """Check if all requirements are met."""
    logger.info("Checking requirements...")

    if not check_cuda_available():
        logger.error("CUDA is not available. This test requires a CUDA-enabled GPU.")
        return False

    # Add more requirement checks here
    logger.info("All requirements met.")
    return True


def run_benchmark(args, logger):
    """Run the actual benchmark."""
    logger.info(f"Starting benchmark for model: {args.model}")
    logger.info(f"Configuration: iterations={args.iterations}, batch_size={args.batch_size}")

    benchmark = GPUBenchmark(
        model_name=args.model,
        batch_size=args.batch_size
    )

    start_time = time.time()

    try:
        results = benchmark.run(iterations=args.iterations)
        elapsed = time.time() - start_time

        logger.info(f"Benchmark completed in {elapsed:.2f} seconds")
        logger.info(f"Average throughput: {results['avg_throughput']:.2f} tokens/sec")
        logger.info(f"Average latency: {results['avg_latency']:.4f} seconds")

        # Save results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        benchmark.save_results(results, args.output)
        logger.info(f"Results saved to {args.output}")

        return results

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return None


def main():
    """Main entry point for the manual test."""
    logger = setup_logging()
    args = parse_args()

    logger.info("=" * 70)
    logger.info("GPU Performance Benchmark - Manual Test")
    logger.info("=" * 70)

    # Check requirements
    if not check_requirements(logger):
        logger.error("Requirements not met. Exiting.")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark(args, logger)

    # Exit with appropriate code
    if results:
        logger.info("Benchmark completed successfully!")
        sys.exit(0)
    else:
        logger.error("Benchmark failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

## Script Structure

### Standard Components

All manual test scripts should include:

1. **Shebang Line**: `#!/usr/bin/env python3`
2. **Docstring**: Purpose, requirements, usage, expected duration
3. **Logging**: Both file and console output
4. **Argument Parsing**: Flexible configuration via CLI
5. **Requirement Checks**: Verify environment before running
6. **Error Handling**: Graceful failure with clear messages
7. **Results Saving**: Store output for later analysis
8. **Exit Codes**: 0 for success, non-zero for failure

### Docstring Template

```python
"""
Brief description of what this manual test does.

This script is not run in CI because [reason].

Requirements:
    - Requirement 1
    - Requirement 2
    - ...

Usage:
    pixi run python tests/manual/manual_script.py
    pixi run python tests/manual/manual_script.py --option value

Environment Variables:
    VAR_NAME: Description (default: value)

Expected Duration:
    ~X minutes/hours for default configuration

Output:
    Description of what output to expect
"""
```

## Best Practices

### Argument Parsing
```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--option', type=str, default='value', help='Help text')
    parser.add_argument('--flag', action='store_true', help='Boolean flag')
    parser.add_argument('--number', type=int, default=10, help='Number value')
    return parser.parse_args()
```

### Logging Setup
```python
import logging

def setup_logging(log_file='manual_test.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
```

### Progress Reporting
```python
from tqdm import tqdm

def run_with_progress(iterations):
    results = []
    for i in tqdm(range(iterations), desc="Running benchmark"):
        result = run_single_iteration(i)
        results.append(result)
    return results
```

### Resource Cleanup
```python
import atexit

def cleanup():
    """Clean up resources on exit."""
    logger.info("Cleaning up resources...")
    # Cleanup code here

# Register cleanup function
atexit.register(cleanup)
```

## Running Manual Tests

```bash
# Run with default settings
pixi run python tests/manual/manual_benchmark.py

# Run with custom arguments
pixi run python tests/manual/manual_benchmark.py --model gpt2 --iterations 100

# Run with environment variables
CUDA_VISIBLE_DEVICES=0,1 pixi run python tests/manual/manual_benchmark.py

# Run and save output to file
pixi run python tests/manual/manual_benchmark.py 2>&1 | tee output.log

# Run in background
nohup pixi run python tests/manual/manual_benchmark.py &

# Make script executable (optional)
chmod +x tests/manual/manual_benchmark.py
./tests/manual/manual_benchmark.py
```

## Common Use Cases

### GPU Benchmarking
```python
"""Benchmark LLM inference on different GPU configurations."""
# Check GPU availability
# Load models with different quantizations
# Measure throughput, latency, memory usage
# Compare results across configurations
```

### Stress Testing
```python
"""Stress test system with prolonged heavy load."""
# Run extended benchmark (hours)
# Monitor memory leaks
# Check thermal throttling
# Verify stability under load
```

### Hardware-Specific Tests
```python
"""Test optimizations for specific hardware platform."""
# Check for specific GPU model
# Test hardware-specific kernels
# Validate performance improvements
# Compare with baseline
```

### Visual Verification
```python
"""Generate visualizations for manual inspection."""
# Create performance plots
# Generate comparison charts
# Output visual diffs
# Save for review
```

### Interactive Tests
```python
"""Interactive test requiring user input."""
# Present options to user
# Collect feedback
# Run selected scenarios
# Generate report
```

## Output and Results

### Save Results
Always save results to the `tmp/` directory:

```python
from pathlib import Path
import json

def save_results(results, filename='results.json'):
    """Save test results to file."""
    output_path = Path('tmp') / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
```

### Generate Reports
```python
def generate_report(results, output_file='tmp/report.md'):
    """Generate markdown report from results."""
    with open(output_file, 'w') as f:
        f.write("# Benchmark Results\n\n")
        f.write(f"## Summary\n\n")
        # Write report content
```

## Guidelines

1. **Prefix with manual_**: Never use `test_` prefix to avoid pytest collection
2. **Document Everything**: Clear docstrings, usage examples, requirements
3. **Make Configurable**: Use argparse for flexibility
4. **Check Requirements**: Verify environment before running expensive operations
5. **Log Progress**: Both to console and file
6. **Save Results**: Always save output to `tmp/` directory
7. **Handle Errors**: Graceful failure with clear error messages
8. **Report Duration**: Include expected runtime in docstring
9. **Exit Codes**: Return meaningful exit codes
10. **Cleanup Resources**: Ensure resources are released even on failure

## Integration with Project

- Results saved to `tmp/` are gitignored
- Can reference results in documentation
- May inform integration/unit test development
- Findings can drive optimization priorities
- Use for validation before releases

## References

- [Python argparse](https://docs.python.org/3/library/argparse.html)
- [Python logging](https://docs.python.org/3/library/logging.html)
- [tqdm Progress Bar](https://github.com/tqdm/tqdm)
