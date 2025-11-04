# NCU Release Scripts

Production-ready scripts for NVIDIA Nsight Compute profiling workflows.
These tools are maintained with care to ensure compatibility with the latest NCU features and LLM architectures.

## Scripts

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
   ./extract-top-kernels.py summary_cuda_gpu_kern_sum.csv -o top-kernels.yaml --topk 5
   ```

4. **Batch profile with Nsight Compute** (kernel-level)
   ```bash
   ../examples/ncu-profile-top-kernels.v2.sh --yaml top-kernels.yaml
   ```

5. **Analyze results**
   - Each kernel gets a directory with:
     - `ncu.ncu-rep` - Open in ncu-ui for interactive analysis
     - `ncu.section_*.csv` - Section data (SpeedOfLight, MemoryWorkloadAnalysis, etc.)
     - `command.yaml` - Profiling provenance

## References

- [Nsight Compute CLI Documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- Example workflows: `../examples/howto-profile-nsys-kernels.md`