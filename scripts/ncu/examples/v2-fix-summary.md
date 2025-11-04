# V2 Scripts - Fix Summary

## What Was Fixed

### Issue 1: Regex Pattern Escaping
The original YAML file had **double-escaped** parentheses in the regex patterns:
```yaml
regex: '.*internal::gemvx::kernel<.*\\(int\\)7.*>'  # WRONG
```

This caused ncu to receive `\\(` which matches a literal backslash + paren, not just a paren.

### Solution: Single Backslash Escaping
Updated to single-escaped parentheses:
```yaml
regex: '.*internal::gemvx::kernel<.*\(int\)7.*>'  # CORRECT
```

Now ncu receives `\(` which correctly matches the literal `(` character in kernel names like `(int)7`.

### Issue 2: Section Collection
Initial v2 scripts only collected individual metrics, but the original script collects comprehensive **sections** which provide richer analysis capabilities.

### Solution: Added Section Collection
Updated v2 scripts now collect the same sections as the original:
- **SpeedOfLight**: High-level compute/memory/L1/L2 utilization percentages
- **MemoryWorkloadAnalysis**: Detailed memory subsystem breakdown
- **Occupancy**: Theoretical vs achieved occupancy metrics
- **SchedulerStats**: Warp scheduling and stall reason analysis

Each section is exported to a separate CSV file for analysis.

## Verification

### Test YAML Parsing
```bash
yq -r '.kernels[].regex' scripts/ncu/examples/top-kernel-example.yaml
```

Expected output:
```
.*internal::gemvx::kernel<.*\(int\)7.*>
.*internal::gemvx::kernel<.*\(int\)6.*>
unrolled_elementwise_kernel<.*direct_copy_kernel_cuda
```

### Test Argument Construction
```bash
./scripts/ncu/examples/ncu-profile-top-kernels.v2.sh --help
```

## Usage

### Profile All Top Kernels (Batch Mode)
```bash
./scripts/ncu/examples/ncu-profile-top-kernels.v2.sh
```

This will:
1. Read kernels from `top-kernel-example.yaml`
2. Profile each kernel with ncu using `--launch-skip 200 --launch-count 1`
3. Output reports to `tmp/ncu-profile-batch-v2/<timestamp>/k{1,2,3}/`

### Profile Single Kernel
```bash
./scripts/ncu/examples/ncu-profile-kernel.v2.sh \
  --kernel-regex '.*internal::gemvx::kernel<.*\(int\)7.*>'
```

### Custom Options
```bash
# Custom YAML file
./scripts/ncu/examples/ncu-profile-top-kernels.v2.sh \
  --yaml my-kernels.yaml

# Custom output directory
./scripts/ncu/examples/ncu-profile-top-kernels.v2.sh \
  --output-root /tmp/my-profiles

# Different sampling parameters
./scripts/ncu/examples/ncu-profile-top-kernels.v2.sh \
  --launch-skip 500 \
  --launch-count 3

# Force overwrite existing reports
./scripts/ncu/examples/ncu-profile-top-kernels.v2.sh \
  --force-overwrite
```

## Expected Results

After running the batch script, you should see:
- ✅ Kernel #1 (gemvx int)7): SUCCESS
- ✅ Kernel #2 (gemvx int)6): SUCCESS
- ✅ Kernel #3 (direct_copy): SUCCESS

Each kernel directory will contain:
- `ncu.ncu-rep` - Main ncu report file (can be opened in ncu-ui)
- `ncu.section_SpeedOfLight.csv` - Speed of Light analysis
- `ncu.section_MemoryWorkloadAnalysis.csv` - Memory subsystem metrics
- `ncu.section_Occupancy.csv` - Occupancy analysis
- `ncu.section_SchedulerStats.csv` - Warp scheduler and stall breakdown
- `ncu.section_SpeedOfLight_RooflineChart.csv` - Roofline model data (if available)
- `ncu.details.csv` - Details page summary
- `command.yaml` - Profiling provenance and configuration

## Key Changes from Original Scripts

1. **Simplified approach**: No discovery phase, no `--kernel-id`
2. **NVIDIA best practices**: Uses `--launch-skip/--launch-count` directly
3. **Regex-based filtering**: Uses `--kernel-name-base demangled` with regex patterns
4. **Section-based collection**: Collects comprehensive sections (SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats) like the original script
5. **Colorized logging**: Better visibility into profiling progress
6. **Smaller codebase**: ~50% reduction in lines of code (380 → 210 lines total)

## Troubleshooting

### "No kernels were profiled"
- Check regex pattern matches kernel name exactly
- Verify single backslashes in YAML (not double)
- Check kernel exists: look at ncu's "Available Kernels" list in output

### Permission errors
- Ensure GPU performance counters are accessible
- May need to run with elevated privileges outside Docker

### Launch skip/count issues
- Increase `--launch-skip` if early invocations are warmup
- Increase `--launch-count` to average multiple samples
- Use NVTX ranges for precise phase targeting (requires code changes)
