create `scripts/ncu/release/ncu-profile-kernels.py`, CLI shape is like this:

```bash
python3 ncu-profile-kernels.py [options] -- <launch-command> [launch-args]
```

**options**:
- `--kernel-config <yaml-path>`: Path to a YAML file listing kernel names and regex patterns to match, like `scripts/ncu/examples/top-kernel-example.yaml`
- `--kernel-regex <regex>`: regex pattern of kernel names to profile, this cannot be used together with `--kernel-yaml`, raise error if both are provided
- `--output-dir <dir>`: Directory to save profiling results and reports, overwrites if exists
- `--extra-sections <section1> <section2> ...`: Additional Nsight Compute sections to collect, apart from default sections
- `--num-kernel-call-skip <N>`: Number of initial kernel invocations to skip for profiling
- `--num-kernel-call-profile <M>`: Number of kernel invocations to profile after skipping

**launch-command**: The command to launch the target application, followed by any necessary arguments