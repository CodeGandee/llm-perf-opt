# NCU Release Scripts

Production-ready scripts for NVIDIA Nsight Compute profiling workflows.
These tools are maintained with care to ensure compatibility with the latest NCU features and LLM architectures.

## Scripts

### ncu-profile-kernels.sh

Bash version of the kernel profiler - recommended for production use as NCU works better when called directly from bash rather than through Python subprocess.

**Usage:**
```bash
ncu-profile-kernels.sh [options] -- <launch-command> [launch-args]
```

**Required (one of):**
- `--kernel-config <yaml-path>` - Path to YAML file with kernel names/regex patterns
- `--kernel-regex <regex>` - Single regex pattern for kernel matching

**Options:**
- `--output-dir <dir>` - Directory for profiling results (default: tmp/ncu-profile/<timestamp>)
- `--topk <num>` - Profile only top K kernels from YAML (requires --kernel-config)
- `--extra-sections <s1> <s2>` - Additional ncu sections beyond defaults
- `--num-kernel-call-skip <N>` - Skip first N kernel invocations (default: 200)
- `--num-kernel-call-profile <M>` - Profile M invocations after skipping (default: 1)
- `--force-overwrite` - Overwrite existing reports

**Default sections:** SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats

**Dependencies:**
- **Required:** `ncu` (Nsight Compute CLI)
- **For --kernel-config mode:**
  - `yq` (preferred, faster) OR `python3/python` with `ruamel.yaml` or `pyyaml`
  - `jq` (preferred for JSON parsing) OR `python3/python`
- **For --kernel-regex mode:** Only `ncu` required (no Python/yq/jq needed)

**Features:**
- Supports both single kernel and batch profiling modes
- Mutually exclusive: `--kernel-config` OR `--kernel-regex`
- `--topk` limits profiling to first K kernels from YAML (only with `--kernel-config`)
- Flexible launch command via `--` separator (can be any executable: Python, compiled binary, bash script, etc.)
- Colored terminal logging
- Automatic CSV export for all sections
- Provenance tracking via `command.yaml`
- NCU runs directly (not through subprocess) for better reliability

**Examples:**
```bash
# Profile single kernel (no Python needed)
ncu-profile-kernels.sh --kernel-regex 'gemvx::kernel<.*\(int\)7.*>' \
  --output-dir tmp/gemvx \
  -- ./my_cuda_app --arg1 --arg2

# Profile multiple kernels from YAML (requires yq or python)
ncu-profile-kernels.sh --kernel-config top-kernels.yaml \
  --extra-sections SourceCounters \
  -- python inference.py --model deepseek

# Profile only top 3 kernels with Pixi environment
ncu-profile-kernels.sh --kernel-config top-kernels.yaml --topk 3 \
  -- pixi run -e rtx5090 python -m llm_perf_opt.runners.llm_profile_runner device=cuda:0

# Profile with C++ binary
ncu-profile-kernels.sh --kernel-regex 'my_kernel.*' \
  -- /path/to/my_cuda_binary input.dat

# Profile with custom sampling
ncu-profile-kernels.sh --kernel-regex 'flash_fwd.*' \
  --num-kernel-call-skip 500 \
  --num-kernel-call-profile 10 \
  -- python benchmark.py
```

**Output structure:**
```
<output-dir>/
├── command.yaml                                # Provenance (timestamp, args, versions)
├── kernel_0001_d41d8cd98f00b204e9800998ecf8427e/  # Per‑kernel dir: rank (min 4 digits) + md5(name)
│   ├── ncu.ncu-rep                              # NCU binary report
│   ├── ncu.section_SpeedOfLight.csv
│   ├── ncu.section_MemoryWorkloadAnalysis.csv
│   ├── ncu.section_Occupancy.csv
│   ├── ncu.section_SchedulerStats.csv
│   └── ncu.details.csv
├── kernel_0002_<md5>/
│   ├── ncu.ncu-rep
│   └── ...
└── kernel_0003_<md5>/
    ├── ncu.ncu-rep
    └── ...
```
Directory name format: `kernel_<rank>_<md5(kernel-name)>`, where `<rank>` is zero‑padded to `max(4, len(total_kernels))`.

**Note:** The directory structure matches the naming pattern from `ncu-profile-kernel.v2.sh`.

### ncu-profile-kernels.py

Python version of the kernel profiler with similar functionality.

**Usage:**
```bash
python3 ncu-profile-kernels.py [options] -- <launch-command> [launch-args]
```

**Required (one of):**
- `--kernel-config <yaml-path>` - Path to YAML file with kernel names/regex patterns
- `--kernel-regex <regex>` - Single regex pattern for kernel matching

**Options:**
- `--output-dir <dir>` - Directory for profiling results (default: tmp/ncu-profile/<timestamp>)
- `--topk <num>` - Profile only top K kernels from YAML (requires --kernel-config)
- `--extra-sections <s1> <s2>` - Additional ncu sections beyond defaults
- `--num-kernel-call-skip <N>` - Skip first N kernel invocations (default: 200)
- `--num-kernel-call-profile <M>` - Profile M invocations after skipping (default: 1)
- `--force-overwrite` - Overwrite existing reports

**Default sections:** SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats

**Features:**
- Supports both single kernel and batch profiling modes
- Mutually exclusive: `--kernel-config` OR `--kernel-regex`
- `--topk` limits profiling to first K kernels from YAML (only with `--kernel-config`)
- Flexible launch command via `--` separator
- Colored terminal logging
- Automatic CSV export for all sections
- Provenance tracking via `command.yaml`

**Examples:**
```bash
# Profile single kernel
python3 ncu-profile-kernels.py \
  --kernel-regex 'internal::gemvx::kernel<.*\(int\)7.*>' \
  --output-dir tmp/gemvx-profile \
  -- python -m llm_perf_opt.runners.llm_profile_runner device=cuda:0

# Profile multiple kernels from YAML
python3 ncu-profile-kernels.py \
  --kernel-config top-kernels.yaml \
  --extra-sections SourceCounters \
  -- python inference.py --model deepseek

# Profile with custom sampling
python3 ncu-profile-kernels.py \
  --kernel-regex 'flash_fwd.*' \
  --num-kernel-call-skip 500 \
  --num-kernel-call-profile 10 \
  -- python benchmark.py

# Profile only top 3 kernels from YAML
python3 ncu-profile-kernels.py \
  --kernel-config top-kernels.yaml \
  --topk 3 \
  -- python inference.py
```

**Output structure:**
```
<output-dir>/
├── command.yaml                                # Provenance (timestamp, args, versions)
├── kernel_0001_d41d8cd98f00b204e9800998ecf8427e/
│   ├── ncu.ncu-rep
│   ├── ncu.section_SpeedOfLight.csv
│   ├── ncu.section_MemoryWorkloadAnalysis.csv
│   ├── ncu.section_Occupancy.csv
│   ├── ncu.section_SchedulerStats.csv
│   └── ncu.details.csv
├── kernel_0002_<md5>/
│   ├── ncu.ncu-rep
│   └── ...
└── kernel_0003_<md5>/
    ├── ncu.ncu-rep
    └── ...
```
Directory name format matches the Bash script.

### test-ncu-profile.sh

Quick test script for `ncu-profile-kernels.sh` with sensible defaults for the RTX 5090 environment.

**Usage:**
```bash
# Run with defaults (top 3 kernels, RTX 5090 env)
./scripts/ncu/release/test-ncu-profile.sh

# Profile top 5 kernels instead
TOPK=5 ./scripts/ncu/release/test-ncu-profile.sh

# Use default Pixi environment instead of RTX 5090
PIXI_ENV=default ./scripts/ncu/release/test-ncu-profile.sh

# Adjust launch skip/count
LAUNCH_SKIP=100 LAUNCH_COUNT=5 ./scripts/ncu/release/test-ncu-profile.sh
```

**Environment variables:**
- `TOPK` - Number of top kernels to profile (default: 3)
- `LAUNCH_SKIP` - Kernel invocations to skip (default: 50)
- `LAUNCH_COUNT` - Kernel invocations to profile (default: 1)
- `PIXI_ENV` - Pixi environment to use (default: rtx5090)
- `MAX_NEW_TOKENS` - Max tokens for inference (default: 64)
- `NUM_SAMPLES` - Number of dataset samples (default: 1)

## Workflow: Profile Top Kernels (NSYS → YAML → NCU)

End‑to‑end steps to profile the top kernels from an Nsight Systems capture:

1) Get a kernel summary CSV from Nsight Systems
- If you have an `.nsys-rep`, export the CUDA kernel summary CSV:
  - `nsys stats -r cuda_gpu_kern_sum -f csv -o summary path/to/your.nsys-rep`
- Or use the sample CSV: `scripts/ncu/examples/summary_cuda_gpu_kern_sum.csv`

2) Generate a kernel config YAML with extract-top-kernels.py
- Example (top 10):
  - `python scripts/ncu/release/extract-top-kernels.py scripts/ncu/examples/summary_cuda_gpu_kern_sum.csv -o tmp/ncu/top-10-kernels.yaml --topk 10`

3) Run Nsight Compute profiling (choose Bash or Python variant)
- Bash (recommended for production):
  - `pixi run -e rtx5090 ./scripts/ncu/release/ncu-profile-kernels.sh --kernel-config tmp/ncu/top-10-kernels.yaml --topk 3 -- python -m llm_perf_opt.runners.llm_profile_runner device=cuda:0`
- Python:
  - `python scripts/ncu/release/ncu-profile-kernels.py --kernel-config tmp/ncu/top-10-kernels.yaml --topk 3 -- ./bin/infer --device cuda:0`

4) Inspect outputs
- Outputs live under `<output-dir>/kernel_<rank>_<md5>/` with files `ncu.ncu-rep`, `ncu.section_<Section>.csv`, and `ncu.details.csv`.
- Default `<output-dir>` is `tmp/ncu-profile/<timestamp>`. Override via `--output-dir`.

Tips
- Add more sections with `--extra-sections SourceCounters`.
- Use the helper test driver: `pixi run -e rtx5090 ./scripts/ncu/release/test-ncu-profile.sh [--bash|--python]`.

### extract-top-kernels.py

Extracts top-K kernels from Nsight Systems CSV output and generates a YAML configuration file for batch profiling with ncu.

**Usage:**
```bash
./extract-top-kernels.py <csv-filepath> --output/-o <path-to-yaml> --topk <k>
```

**Features:**
- Reads nsys kernel summary CSV (e.g., `summary_cuda_gpu_kern_sum.csv`)
- Sorts kernels by time percentage or total time (nanoseconds)
- Generates exact-match regex patterns using `re.escape()` (NVIDIA best practices)
- Outputs YAML compatible with `ncu-profile-top-kernels.v2.sh`

**Examples:**
```bash
# Extract all kernels sorted by time % (default behavior)
./extract-top-kernels.py summary_cuda_gpu_kern_sum.csv -o all-kernels.yaml

# Extract top-3 kernels sorted by time %
./extract-top-kernels.py summary_cuda_gpu_kern_sum.csv -o top-3-kernels.yaml --topk 3

# Extract top-5 kernels sorted by total time (ns)
./extract-top-kernels.py summary.csv -o top-5.yaml --topk 5 --sort-by time_ns
```

**Generated YAML format:**
```yaml
kernels:
  - name: 'full::demangled::kernel<name>'
    regex: '^escaped\\ regex\\ pattern$'
    description: 'Kernel #1 - 30.0% of GPU time, 51146 calls'
```

**Note on regex patterns:**
- The script generates **exact-match patterns** using `re.escape()` for safety
- You may want to manually edit patterns to match kernel **families** instead
- Example: Change `^internal::gemvx::kernel<...\(int\)7...>$` to `.*internal::gemvx::kernel<.*\(int\)7.*>` to match all `(int)7` variants

**Next steps after generation:**
1. Review the generated YAML and adjust regex patterns if needed
2. Run batch profiling: `../examples/ncu-profile-top-kernels.v2.sh --yaml <output.yaml>`

## Workflow

### End-to-End Kernel Profiling Pipeline

#### Option A: Using ncu-profile-kernels.py (Recommended)

1. **Profile with Nsight Systems** (application-level)
   ```bash
   nsys profile -o myapp.nsys-rep python run.py
   ```

2. **Export kernel summary CSV**
   ```bash
   nsys stats --report cuda_gpu_kern_sum --format csv --output . myapp.nsys-rep
   # Generates: summary_cuda_gpu_kern_sum.csv
   ```

3. **Extract top kernels**
   ```bash
   python3 extract-top-kernels.py summary_cuda_gpu_kern_sum.csv -o top-kernels.yaml --topk 5
   ```

4. **Batch profile with Nsight Compute** (kernel-level)
   ```bash
   python3 ncu-profile-kernels.py \
     --kernel-config top-kernels.yaml \
     --output-dir tmp/ncu-batch \
     -- python run.py
   ```

   Or profile only the top 3 most expensive kernels:
   ```bash
   python3 ncu-profile-kernels.py \
     --kernel-config top-kernels.yaml \
     --topk 3 \
     --output-dir tmp/ncu-top3 \
     -- python run.py
   ```

5. **Analyze results**
   - Output directory contains:
     - `kernel_NNN_<name>.ncu-rep` - Open in ncu-ui for interactive analysis
     - `kernel_NNN_<name>.section_*.csv` - Section data (SpeedOfLight, MemoryWorkloadAnalysis, etc.)
     - `kernel_NNN_<name>.details.csv` - Detailed metrics
     - `command.yaml` - Profiling provenance

#### Option B: Single Kernel Quick Profile

For quick profiling of a specific kernel pattern:

```bash
python3 ncu-profile-kernels.py \
  --kernel-regex 'your_kernel_pattern.*' \
  --num-kernel-call-skip 100 \
  -- python run.py --args
```

#### Option C: Using Legacy Bash Script

For backward compatibility:

```bash
../examples/ncu-profile-top-kernels.v2.sh --yaml top-kernels.yaml
```

## References

- [Nsight Compute CLI Documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- Example workflows: `../examples/howto-profile-nsys-kernels.md`
