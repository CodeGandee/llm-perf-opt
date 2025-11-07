# Code Review: NCU Profiling Infrastructure - Roofline Data Collection Issue

**Date**: 2025-11-07 12:11:32
**Scope**: NCU profiling scripts and configuration management
**Reviewer**: Claude Code
**Issue**: Roofline plots show normalized throughput (DRAM% vs SM%) instead of physical roofline (FLOPs/Byte vs FLOPs/sec)

---

## Executive Summary

The NCU profiling infrastructure has a **configuration maintenance issue**: Standalone profiling scripts (`ncu-profile-kernels.sh` and `.py`) use hardcoded section lists that exclude roofline sections, while Hydra configurations correctly specify roofline collection via `set: roofline`. The scripts are **intentionally independent** of the Hydra system (for users who want raw NCU CLI tools), but currently duplicate configuration instead of reading Hydra config YAMLs as plain files for single-source-of-truth.

This results in empty roofline CSV files and prevents proper roofline analysis.

**Impact**: ⚠️ **HIGH** - Missing critical performance analysis data (arithmetic intensity, FLOP rates)
**Effort to Fix**: 🟢 **LOW** - Simple configuration change
**Design Intent**: ✅ **Correct** - Scripts are intentionally Hydra-independent, but should maintain config consistency via plain YAML reading

---

## 1. Root Cause Analysis

### 1.1 The Missing Roofline Data

**Observation**: All `SpeedOfLight_RooflineChart.csv` files are **empty (0 bytes)**

```bash
$ ls -lh tmp/ncu-profile/20251107-105933/kernel_0001_*/ncu.section_SpeedOfLight_RooflineChart.csv
-rw-rw-r-- 1 igame igame 0 Nov  7 10:59 ...RooflineChart.csv  # EMPTY!
```

**Why**: The `SpeedOfLight_RooflineChart` section is not collected during profiling.

---

### 1.2 Hardcoded Sections in Standalone Scripts

**Bash script** (`scripts/ncu/release/ncu-profile-kernels.sh:97`):
```bash
DEFAULT_SECTIONS="SpeedOfLight MemoryWorkloadAnalysis Occupancy SchedulerStats"
```

**Python script** (`scripts/ncu/release/ncu-profile-kernels.py:528`):
```python
default_sections = ["SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy", "SchedulerStats"]
```

❌ **Missing**: `SpeedOfLight_RooflineChart`

**Impact**: When scripts build NCU command (line 451-453 in bash, similar in Python):
```bash
for section in $ALL_SECTIONS; do
  NCU_ARGS+=(--section "$section")
done
```

The roofline section is never included → NCU doesn't collect the data → CSV export is empty.

---

### 1.3 Hydra Configurations Are Correct

**Hydra config** (`conf/profiling/ncu/ncu.default.yaml:20-21`):
```yaml
ncu_cli:
  set: roofline  # ✓ Correct! This NCU preset includes roofline sections
```

**What `set: roofline` includes** (verified with `ncu --list-sets`):
```
roofline preset sections:
  - SpeedOfLight
  - SpeedOfLight_RooflineChart                            ← THE MISSING SECTION
  - SpeedOfLight_HierarchicalDoubleRooflineChart
  - SpeedOfLight_HierarchicalHalfRooflineChart
  - SpeedOfLight_HierarchicalSingleRooflineChart
  - SpeedOfLight_HierarchicalTensorRooflineChart
  - WorkloadDistribution
```

**The Issue**: Standalone scripts **don't read Hydra config YAMLs** (intentional independence from Hydra system, but creates config duplication):
```bash
$ grep -r "ruamel.yaml.*conf/profiling\|yaml.load.*ncu" scripts/ncu/release/*.py
# No matches found - scripts use hardcoded defaults instead
```

---

### 1.4 Intentional Design: Dual NCU Profiling Paths

The project has **two independent NCU profiling approaches** (by design):

#### Path 1: Standalone Scripts (Raw NCU CLI Tools)
```bash
# Direct invocation - for users who want raw ncu without Hydra
scripts/ncu/release/ncu-profile-kernels.sh \
  --kernel-config top-10-kernels.yaml \
  --topk 10 \
  -- python -m llm_perf_opt.runners.llm_profile_runner ...
```
- ✅ **Intentionally Hydra-independent** (raw NCU CLI wrapper)
- ❌ Currently uses hardcoded sections (config duplication)
- ❌ Missing roofline data
- 🔧 **Should**: Read sections from `conf/profiling/ncu/*.yaml` as plain YAML files

#### Path 2: Hydra-Based Runners (Integrated Workflow)
```bash
# Via deep_profile_runner - full Hydra integration
python -m llm_perf_opt.runners.deep_profile_runner \
  hydra.run.dir=... \
  pipeline.ncu.enable=true \
  profiling@pipeline.ncu=ncu/ncu.default
```
- ✅ Uses Hydra system (OmegaConf composition)
- ✅ Correct `set: roofline` preset
- ✅ Collects roofline data correctly

**Design Rationale**: Standalone scripts serve users who want lightweight NCU profiling without the full Hydra ecosystem. However, they should still maintain **single-source-of-truth** by reading config YAMLs directly (not using Hydra system).

---

## 2. Impact Assessment

### 2.1 Missing Analysis Capabilities

Without roofline CSV data, the analysis script **cannot produce**:
1. **Physical Roofline Plot**: Arithmetic Intensity (FLOPs/Byte) vs Performance (FLOPs/sec)
2. **Hierarchical Roofline Plots**: FP16/FP32/FP64/Tensor precision breakdowns
3. **Bottleneck Classification**: Memory-bound vs compute-bound based on roofline position

**Current Fallback** (`scripts/ncu/analysis/analyze_ncu_dir.py:553-571`):
```python
# Normalized plot when roofline data unavailable
plot_df = summary_df.dropna(subset=["dram_throughput_pct", "sm_throughput_pct"])
ax.scatter(plot_df["dram_throughput_pct"], plot_df["sm_throughput_pct"])
```
✅ **Works** but only shows relative throughput, not absolute FLOPs

---

### 2.2 Downstream Tool Compatibility

Missing roofline data prevents integration with:
- NVIDIA Nsight Compute GUI (can't import for visual analysis)
- Automated optimization workflows that depend on AI/performance metrics
- Comparative analysis across GPU architectures

---

## 3. Why This Happened

### 3.1 Design Context

Based on code structure and comments:

**Standalone Scripts Intent**:
- Designed as lightweight, Hydra-independent NCU CLI wrappers
- Target users: Those who want raw `ncu` functionality without full project infrastructure
- No dependencies on Hydra system (OmegaConf, Hydra composition, @hydra.main decorators)

**Configuration Evolution**:
1. **Phase 1**: Scripts created with hardcoded defaults (minimal, common sections)
2. **Phase 2**: Hydra configs added for integrated workflows
3. **Current**: Two parallel configs that should be unified via plain YAML reading

Evidence: Comment in `ncu-profile-kernels.sh:33-34`:
```bash
# Default sections:
#   SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats
```
These match PyTorch Profiler's common metrics but predate roofline support addition to Hydra configs.

---

### 3.2 Configuration Duplication

Current state has **2 independent sources of truth**:
1. **Standalone scripts**: Bash/Python hardcoded `DEFAULT_SECTIONS`
2. **Hydra configs**: `conf/profiling/ncu/*.yaml` with `ncu_cli.sections` + `set` preset

**Problem**: These must be manually kept in sync → maintenance burden and drift.

**Solution**: Maintain single-source-of-truth by having scripts read Hydra YAML files directly (as plain YAML, not via Hydra system).

---

## 4. Recommended Solutions

### Solution 1: Quick Fix - Update Hardcoded Defaults (Immediate)

**Priority**: 🔴 **HIGH** - Minimal code change, maximum impact

**Change 1**: Bash script (`scripts/ncu/release/ncu-profile-kernels.sh:97`)
```bash
# BEFORE:
DEFAULT_SECTIONS="SpeedOfLight MemoryWorkloadAnalysis Occupancy SchedulerStats"

# AFTER:
DEFAULT_SECTIONS="SpeedOfLight MemoryWorkloadAnalysis Occupancy SchedulerStats SpeedOfLight_RooflineChart"
```

**Change 2**: Python script (`scripts/ncu/release/ncu-profile-kernels.py:528`)
```python
# BEFORE:
default_sections = ["SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy", "SchedulerStats"]

# AFTER:
default_sections = [
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "Occupancy",
    "SchedulerStats",
    "SpeedOfLight_RooflineChart"  # Enable roofline analysis
]
```

**Testing**:
```bash
# Re-run profiling
pixi run -e rtx5090 scripts/ncu/release/test-ncu-profile.sh --bash

# Verify roofline CSV is populated
ls -lh tmp/ncu-profile/*/kernel_*/ncu.section_SpeedOfLight_RooflineChart.csv
# Should show non-zero file sizes

# Run analysis
pixi run python scripts/ncu/analysis/analyze_ncu_dir.py tmp/ncu-profile/<run_id>/

# Check for physical roofline plot
ls -lh tmp/ncu-profile/<run_id>/analysis/roofline_scatter.png
# Should exist (not roofline_scatter_normalized.png)
```

**Trade-offs**:
- ✅ **Pro**: Fixes issue immediately for all users
- ✅ **Pro**: No API changes, backward compatible
- ❌ **Con**: Doesn't address root architectural issue

---

### Solution 2: Use NCU Preset Instead of Explicit Sections (Better)

**Priority**: 🟡 **MEDIUM** - Cleaner, leverages NCU's built-in presets

**Rationale**: Instead of listing sections explicitly, use `--set roofline` which includes all roofline sections automatically.

**Change**: Bash script (`scripts/ncu/release/ncu-profile-kernels.sh:441-453`)
```bash
# BEFORE:
NCU_ARGS=(
  --kernel-name-base demangled
  --kernel-name "regex:$KERNEL_REGEX_CURRENT"
  --launch-skip "$LAUNCH_SKIP"
  --launch-count "$LAUNCH_COUNT"
  -o "$OUTPUT_BASE"
)

# Add sections
for section in $ALL_SECTIONS; do
  NCU_ARGS+=(--section "$section")
done

# AFTER:
NCU_ARGS=(
  --kernel-name-base demangled
  --kernel-name "regex:$KERNEL_REGEX_CURRENT"
  --launch-skip "$LAUNCH_SKIP"
  --launch-count "$LAUNCH_COUNT"
  --set roofline   # Use NCU preset instead of explicit sections
  -o "$OUTPUT_BASE"
)

# Add extra sections if specified (on top of preset)
for section in ${EXTRA_SECTIONS[@]}; do
  NCU_ARGS+=(--section "$section")
done
```

**Benefits**:
1. **Future-proof**: Automatically includes new roofline sections added by NVIDIA
2. **Simpler**: Single `--set roofline` vs listing 7+ sections
3. **Consistent**: Matches Hydra config (`set: roofline`)

**Considerations**:
- ⚠️ `--set roofline` collects **more** data (all hierarchical roofline variants)
- Profiling overhead: ~5-10% slower, larger report files
- **Recommendation**: Make this configurable via `--preset` flag (default: `roofline`)

---

### Solution 3: Read Hydra Configs as Plain YAML (Single-Source-of-Truth Fix)

**Priority**: 🟡 **MEDIUM** - Maintains independence while eliminating duplication

**Goal**: Allow scripts to optionally read sections from Hydra YAML configs via CLI flag, maintaining single-source-of-truth while keeping Hydra-independence and backward compatibility.

**Design**: Add `--ncu-config <path>` flag to both scripts. If not provided, use hardcoded defaults.

---

#### For Bash Script (`ncu-profile-kernels.sh`)

**Add CLI argument**:
```bash
# In argument parsing section (after --extra-sections)
NCU_CONFIG_FILE=""  # Empty by default

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ncu-config)
      NCU_CONFIG_FILE="$2"
      shift 2
      ;;
    --extra-sections)
      shift
      while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
        EXTRA_SECTIONS+=("$1")
        shift
      done
      ;;
    # ... other arguments
  esac
done
```

**Load sections function** (using `yq`):
```bash
# Function to load sections from NCU config YAML (plain YAML reading, not Hydra system)
load_ncu_sections_from_yaml() {
  local config_file="$1"

  # Return empty if no config file specified (caller will use hardcoded defaults)
  if [[ -z "$config_file" ]]; then
    return
  fi

  if [[ ! -f "$config_file" ]]; then
    log_warn "NCU config YAML not found: $config_file, using hardcoded defaults"
    return
  fi

  # Check if yq is available (mikefarah's yq, Go-based)
  if ! command -v yq &>/dev/null; then
    log_warn "yq not found, cannot read config YAML. Using hardcoded defaults"
    log_warn "Install yq: https://github.com/mikefarah/yq"
    return
  fi

  # Read sections from ncu_cli.sections array
  local sections
  sections=$(yq eval '.ncu_cli.sections[]' "$config_file" 2>/dev/null | tr '\n' ' ')

  # If sections array exists and non-empty, use it
  if [[ -n "$sections" ]]; then
    echo "$sections"
    return
  fi

  # Fallback: check if 'set' preset is specified
  local preset
  preset=$(yq eval '.ncu_cli.set' "$config_file" 2>/dev/null)
  if [[ "$preset" == "roofline" ]]; then
    # Use roofline preset sections (matches NCU --set roofline)
    echo "SpeedOfLight MemoryWorkloadAnalysis Occupancy SchedulerStats SpeedOfLight_RooflineChart WorkloadDistribution"
    return
  fi

  log_warn "No sections or preset found in $config_file, using hardcoded defaults"
}

# Set default sections
HARDCODED_SECTIONS="SpeedOfLight MemoryWorkloadAnalysis Occupancy SchedulerStats"

# Try to load from config if specified
if [[ -n "$NCU_CONFIG_FILE" ]]; then
  CONFIG_SECTIONS=$(load_ncu_sections_from_yaml "$NCU_CONFIG_FILE")
  if [[ -n "$CONFIG_SECTIONS" ]]; then
    DEFAULT_SECTIONS="$CONFIG_SECTIONS"
    log_info "Loaded sections from config: $(log_highlight "$NCU_CONFIG_FILE")"
  else
    DEFAULT_SECTIONS="$HARDCODED_SECTIONS"
    log_info "Using hardcoded default sections"
  fi
else
  DEFAULT_SECTIONS="$HARDCODED_SECTIONS"
  log_info "No --ncu-config specified, using hardcoded defaults"
fi
```

---

#### For Python Script (`ncu-profile-kernels.py`)

**Add CLI argument**:
```python
# In argparse setup (around line 470)
parser.add_argument(
    "--ncu-config",
    type=str,
    default=None,
    metavar="PATH",
    help="Path to NCU config YAML (e.g., conf/profiling/ncu/ncu.default.yaml). "
         "If not specified, uses hardcoded defaults."
)
```

**Load sections function** (using `ruamel.yaml`):
```python
def load_ncu_sections_from_config_yaml(config_path: str | None) -> dict | None:
    """
    Load NCU configuration from YAML file (plain YAML reading, not Hydra system).

    This maintains single-source-of-truth without depending on Hydra system.
    Scripts remain standalone and Hydra-independent.

    Parameters
    ----------
    config_path : str | None
        Path to NCU config YAML file, or None to skip loading

    Returns
    -------
    dict | None
        Config dict with 'sections', 'set', 'metrics', or None if loading failed/skipped
    """
    if config_path is None:
        return None

    config_file = Path(config_path)
    if not config_file.exists():
        log.warn(f"NCU config YAML not found: {config_file}")
        log.warn("Using hardcoded default sections")
        return None

    # Use existing ruamel.yaml (no new dependencies)
    yaml = ruamel.yaml.YAML(typ="safe")
    try:
        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.load(f)
    except Exception as e:
        log.warn(f"Failed to parse YAML {config_file}: {e}")
        log.warn("Using hardcoded default sections")
        return None

    ncu_cli = cfg.get("ncu_cli", {})
    if not ncu_cli:
        log.warn(f"No 'ncu_cli' section found in {config_file}")
        log.warn("Using hardcoded default sections")
        return None

    # Get sections array
    sections = ncu_cli.get("sections", [])

    # If no explicit sections but 'set' preset is specified, derive sections
    preset = ncu_cli.get("set")
    if not sections and preset == "roofline":
        # Use roofline preset sections (matches NCU --set roofline)
        sections = [
            "SpeedOfLight",
            "MemoryWorkloadAnalysis",
            "Occupancy",
            "SchedulerStats",
            "SpeedOfLight_RooflineChart",
            "WorkloadDistribution"
        ]
        log.info(f"Derived sections from preset: {preset}")

    return {
        "sections": sections,
        "set": preset,
        "metrics": ncu_cli.get("metrics", [])
    }


# In main() (around line 525):
# Hardcoded defaults
hardcoded_sections = ["SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy", "SchedulerStats"]

# Try to load from config if specified
if args.ncu_config:
    ncu_config = load_ncu_sections_from_config_yaml(args.ncu_config)
    if ncu_config:
        sections = ncu_config.get("sections", hardcoded_sections)
        ncu_preset = ncu_config.get("set")
        log.info(f"Loaded sections from config: {args.ncu_config}")
        log.info(f"  Sections: {', '.join(sections)}")
        if ncu_preset:
            log.info(f"  Preset: {ncu_preset}")
    else:
        sections = hardcoded_sections
        log.info("Using hardcoded default sections")
else:
    sections = hardcoded_sections
    log.info("No --ncu-config specified, using hardcoded defaults")

# Add extra sections if specified
if args.extra_sections:
    sections.extend(args.extra_sections)
    log.info(f"Added extra sections: {', '.join(args.extra_sections)}")
```

---

**Updated CLI Integration**:

```bash
# Default: Use hardcoded sections (backward compatible)
scripts/ncu/release/ncu-profile-kernels.sh \
  --kernel-config top-10.yaml \
  -- <launch-command>

# With NCU config: Read sections from Hydra config YAML
scripts/ncu/release/ncu-profile-kernels.sh \
  --kernel-config top-10.yaml \
  --ncu-config conf/profiling/ncu/ncu.default.yaml \
  -- <launch-command>

# With custom config: User-provided YAML
scripts/ncu/release/ncu-profile-kernels.sh \
  --kernel-config top-10.yaml \
  --ncu-config /path/to/my-custom-ncu-config.yaml \
  -- <launch-command>

# Override sections from config with extra sections
scripts/ncu/release/ncu-profile-kernels.sh \
  --kernel-config top-10.yaml \
  --ncu-config conf/profiling/ncu/ncu.default.yaml \
  --extra-sections SourceCounters \
  -- <launch-command>

# Python script - same usage
python3 scripts/ncu/release/ncu-profile-kernels.py \
  --kernel-config top-10.yaml \
  --ncu-config conf/profiling/ncu/ncu.high.yaml \
  -- <launch-command>
```

---

**Integration with Test Script**:

Update `scripts/ncu/release/test-ncu-profile.sh` to use config:

```bash
# In test-ncu-profile.sh (around line 84):
if [[ "$BRANCH" == "python" ]]; then
  python scripts/ncu/release/ncu-profile-kernels.py \
    --kernel-config "$KERNEL_CONFIG" \
    --topk "$TOPK" \
    --output-dir "$OUTPUT_DIR" \
    --ncu-config conf/profiling/ncu/ncu.default.yaml \  # ADD THIS LINE
    --num-kernel-call-skip "$LAUNCH_SKIP" \
    --num-kernel-call-profile "$LAUNCH_COUNT" \
    -- <rest-of-command>
else
  scripts/ncu/release/ncu-profile-kernels.sh \
    --kernel-config "$KERNEL_CONFIG" \
    --topk "$TOPK" \
    --output-dir "$OUTPUT_DIR" \
    --ncu-config conf/profiling/ncu/ncu.default.yaml \  # ADD THIS LINE
    --num-kernel-call-skip "$LAUNCH_SKIP" \
    --num-kernel-call-profile "$LAUNCH_COUNT" \
    -- <rest-of-command>
fi
```

---

**Benefits**:
- ✅ **Backward compatible**: Works without `--ncu-config` (uses hardcoded defaults)
- ✅ **Flexible**: Users can specify any config file path
- ✅ **No assumptions**: Doesn't assume repo structure or script location
- ✅ **Explicit**: Clear when using config vs hardcoded defaults
- ✅ **Single source of truth**: Can use Hydra configs when desired
- ✅ **No new dependencies**:
  - Bash: Uses `yq` (already in codebase, graceful fallback if unavailable)
  - Python: Uses `ruamel.yaml` (already imported)
- ✅ **Scripts remain Hydra-independent** (no OmegaConf, no @hydra.main)
- ✅ **Works with any YAML**: Not limited to Hydra config structure

**Challenges**:
- 🔶 Users must manually specify `--ncu-config` to use Hydra configs
  - **Mitigation**: Update test scripts and documentation to show best practice
- 🔶 Bash script requires `yq` to read config
  - **Mitigation**: Graceful fallback to hardcoded defaults if unavailable
- 🔶 No automatic discovery of config files
  - **Mitigation**: This is intentional - explicit is better than implicit

**Recommendation**:
- **Immediate**: Implement this approach (backward compatible, explicit)
- **Documentation**:
  - Update script help text to mention `--ncu-config` flag
  - Show examples in README of using Hydra configs
  - Update test scripts to use `--ncu-config conf/profiling/ncu/ncu.default.yaml`
- **Validation**: Add CI check that scripts work both with and without `--ncu-config`

---

## 5. Additional Findings & Best Practices

### 5.1 Section Export Post-Processing

**Current code** (`scripts/ncu/release/ncu-profile-kernels.sh:483-490`):
```bash
# Try SpeedOfLight_RooflineChart
ROOFLINE_CSV="${OUTPUT_BASE}.section_SpeedOfLight_RooflineChart.csv"
if ncu --csv --section SpeedOfLight_RooflineChart --import "$REPORT_PATH" > "$ROOFLINE_CSV" 2>/dev/null; then
  [[ -s "$ROOFLINE_CSV" ]] && log_ok "Section CSV: $ROOFLINE_CSV"
fi
```

**Issue**: This **post-export** works only if the section was collected during profiling. Cannot extract data that wasn't captured.

**Misunderstanding** (inferred from code structure):
Developer may have assumed `--section <name> --import` could extract sections retroactively. This is false.

**Correct NCU Behavior**:
1. **Profiling time**: `ncu --section X --set Y -o report.ncu-rep <app>` → collects data
2. **Export time**: `ncu --csv --section X --import report.ncu-rep` → exports **already collected** data

**Recommendation**: Document this clearly in script comments to prevent future confusion.

---

### 5.2 NCU Preset vs Explicit Sections

**When to use presets** (`--set <name>`):
- ✅ General-purpose profiling (roofline, full, detailed)
- ✅ Reproducible workflows across users
- ✅ Avoiding manual section list maintenance

**When to use explicit sections** (`--section <name>`):
- ✅ Targeted analysis (only specific sections needed)
- ✅ Minimizing profiling overhead
- ✅ Custom section combinations not in any preset

**Current scripts**: Use explicit sections but should use preset for roofline analysis.

---

### 5.3 Hydra Config Design Quality

**Assessment**: ✅ **Good**

The Hydra configs (`conf/profiling/ncu/*.yaml`) are well-designed:
1. Clear structure with `ncu_cli` namespace
2. Comprehensive comments explaining each option
3. Proper use of NCU preset (`set: roofline`)
4. Separation of concerns (default, high, device-specific presets)

**Only issue**: Not being used by standalone scripts!

---

### 5.4 Analysis Script Robustness

**Code**: `scripts/ncu/analysis/analyze_ncu_dir.py:284-346`

**Assessment**: ✅ **Excellent**

The roofline parsing logic is robust:
1. Gracefully handles missing/empty CSV files
2. Heuristic column name matching (version-resilient)
3. Unit normalization (TFLOP/s → FLOP/s)
4. Proper fallback to normalized plot

**No changes needed** - the analysis code is doing exactly what it should.

---

## 6. Testing & Validation

### 6.1 Recommended Test Plan

After implementing any solution:

**Test 1: Verify roofline data collection**
```bash
# Run profiling
pixi run -e rtx5090 scripts/ncu/release/test-ncu-profile.sh

# Check CSV file sizes (should be non-zero)
find tmp/ncu-profile -name "*.section_SpeedOfLight_RooflineChart.csv" -exec ls -lh {} \;

# Verify CSV content
head -5 tmp/ncu-profile/*/kernel_0001_*/ncu.section_SpeedOfLight_RooflineChart.csv
# Should show headers like "Arithmetic Intensity (FLOP/Byte)" and "Performance (GFLOP/sec)"
```

**Test 2: Verify analysis output**
```bash
# Run analysis
pixi run python scripts/ncu/analysis/analyze_ncu_dir.py tmp/ncu-profile/<run_id>/

# Check for physical roofline plot (not normalized fallback)
ls -lh tmp/ncu-profile/<run_id>/analysis/roofline_scatter.png
# Should exist with reasonable file size (>50KB)

# Inspect plot visually
# Axes should be: X="Arithmetic Intensity (FLOPs/Byte)", Y="Performance (FLOPs/s)"
# Both axes should be log-scale
```

**Test 3: Verify metrics in summary CSV**
```bash
# Check if roofline metrics are populated
head -1 tmp/ncu-profile/<run_id>/analysis/metrics_summary.csv | tr ',' '\n' | grep roofline
# Expected output:
# roofline_ai_flops_per_byte
# roofline_flops_per_s

# Check data rows (should have non-null values)
csvtool col 22,23 tmp/ncu-profile/<run_id>/analysis/metrics_summary.csv | head -5
```

---

### 6.2 Regression Testing

**Ensure backward compatibility**:
1. ✅ Existing `--extra-sections` flag still works
2. ✅ Scripts run successfully without Hydra configs (fallback to hardcoded)
3. ✅ CSV export for all sections still generates correctly
4. ✅ Analysis script handles both old (no roofline) and new (with roofline) outputs

---

## 7. Implementation Recommendations

### Priority Order

#### Immediate (Sprint 1)
1. **Update hardcoded sections** to include `SpeedOfLight_RooflineChart` (Solution 1)
   - Effort: 5 minutes
   - Impact: HIGH
   - Risk: VERY LOW

2. **Document NCU section behavior** in script comments
   - Clarify that post-export requires prior collection
   - Add examples of `--set roofline` usage

#### Short-term (Sprint 2-3)
3. **Switch to `--set roofline` preset** (Solution 2)
   - Effort: 1 hour (including testing)
   - Impact: MEDIUM-HIGH
   - Risk: LOW
   - Add `--preset` CLI flag for flexibility

4. **Update test scripts** to verify roofline collection
   - Add validation step in `test-ncu-profile.sh`
   - Create assertion for non-empty CSV files

#### Medium-term (Sprint 4-6)
5. **Read Hydra config YAMLs directly** (Solution 3)
   - Effort: 2-4 hours
   - Impact: MEDIUM-HIGH (maintainability, single-source-of-truth)
   - Risk: LOW (plain YAML reading, no Hydra system dependency)
   - Maintains script independence while eliminating config duplication

6. **Update documentation**
   - Clarify that scripts are Hydra-independent by design
   - Document how they maintain single-source-of-truth via plain YAML reading
   - Explain when to use standalone scripts vs Hydra runners

---

## 8. Documentation Updates Needed

### 8.1 Script Documentation

**File**: `scripts/ncu/release/ncu-profile-kernels.sh` (lines 33-34)

**BEFORE**:
```bash
# Default sections:
#   SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats
```

**AFTER**:
```bash
# Default sections (aligned with Hydra conf/profiling/ncu/ncu.default.yaml):
#   SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, SchedulerStats, SpeedOfLight_RooflineChart
#
# Note: SpeedOfLight_RooflineChart is required for physical roofline analysis
# (arithmetic intensity vs FLOPs/sec). Without it, only normalized throughput
# plots are available (DRAM% vs SM%).
#
# Alternative: Use '--set roofline' NCU preset which includes all roofline sections
# plus hierarchical precision variants (FP16/FP32/FP64/Tensor).
```

---

### 8.2 README Updates

**File**: `scripts/ncu/README.md` (create if doesn't exist)

Add section:
```markdown
## Roofline Analysis

To enable roofline plots (arithmetic intensity vs performance):

### Option 1: Using built-in presets (recommended)
```bash
ncu-profile-kernels.sh \
  --kernel-config top-10.yaml \
  --preset roofline \  # Uses NCU roofline preset
  -- <launch-command>
```

### Option 2: Explicit sections
```bash
ncu-profile-kernels.sh \
  --kernel-config top-10.yaml \
  --extra-sections SpeedOfLight_RooflineChart \
  -- <launch-command>
```

### Verification
After profiling, check that roofline CSVs are non-empty:
```bash
find tmp/ncu-profile -name "*RooflineChart.csv" -exec wc -l {} \;
# Should show >1 lines (header + data rows)
```

The analysis script will automatically generate physical roofline plots
when this data is available.
```

---

### 8.3 CLAUDE.md Updates

**File**: `CLAUDE.md` (lines covering NCU profiling)

Add note:
```markdown
### NCU Roofline Collection

**Important**: To collect roofline data for physical roofline analysis:
- Use `--extra-sections SpeedOfLight_RooflineChart` with standalone scripts, OR
- Use `profiling@pipeline.ncu=ncu/ncu.default` which uses `set: roofline` preset

Without roofline sections, analysis falls back to normalized throughput plots.
```

---

## 9. References & Resources

### NVIDIA Documentation
- [Nsight Compute CLI User Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) - Section presets and profiling modes
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) - Best practices for kernel profiling

### Internal Project Files
- `conf/profiling/ncu/ncu.default.yaml` - Correct Hydra configuration (uses `set: roofline`)
- `scripts/ncu/analysis/analyze_ncu_dir.py:284-346` - Roofline CSV parsing logic
- `scripts/ncu/release/ncu-profile-kernels.sh:97` - Hardcoded sections (missing roofline)
- `scripts/ncu/release/ncu-profile-kernels.py:528` - Python equivalent

### Verified NCU Presets
```bash
$ ncu --list-sets | grep roofline -A 3
roofline   SpeedOfLight, SpeedOfLight_HierarchicalDoubleRooflineChart,
           SpeedOfLight_HierarchicalHalfRooflineChart,
           SpeedOfLight_HierarchicalSingleRooflineChart,
           SpeedOfLight_HierarchicalTensorRooflineChart,
           SpeedOfLight_RooflineChart, WorkloadDistribution
```

---

## 10. Conclusion

### Summary of Issues
1. 🔴 **Critical**: Roofline CSV files are empty because `SpeedOfLight_RooflineChart` section not collected
2. 🟡 **Moderate**: Configuration duplication - scripts use hardcoded sections instead of reading Hydra YAML configs
3. 🟢 **Minor**: Design intent (Hydra-independent scripts) not well-documented

### Impact
- **User Experience**: Missing valuable roofline analysis capability
- **Data Quality**: Only relative throughput metrics available, not absolute FLOPs
- **Maintainability**: Two sources of truth (hardcoded + Hydra YAMLs) must be kept in sync manually

### Recommended Action
**Immediate**: Add `SpeedOfLight_RooflineChart` to default sections (5-minute fix)
**Short-term**: Switch to `--set roofline` preset for consistency
**Medium-term**: Read Hydra config YAMLs directly (plain YAML, not Hydra system) for single-source-of-truth

### Confidence Level
**HIGH** - Root cause verified through:
- Direct inspection of NCU output files (empty CSVs)
- Code review of profiling scripts (missing sections)
- NCU preset verification (`ncu --list-sets`)
- Hydra config analysis (correct but unused)

---

**Review completed**: 2025-11-07 12:11:32
**Next steps**: Implement Solution 1 (immediate fix), then evaluate Solutions 2-3 for architectural improvements
