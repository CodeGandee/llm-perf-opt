## Context

Wan2.1 analytic sizing currently has a narrow pipeline:

- [run_ngu800p_concurrency_sweep.py](/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/sizing/run_ngu800p_concurrency_sweep.py) imports `NGU800P` directly, hard-codes NGU-specific output paths, and writes metadata with `device.name = "NGU800P"`.
- [run_make_ngu800p_sweep_figures.py](/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/scripts/reporting/run_make_ngu800p_sweep_figures.py) assumes NGU naming, NGU plot titles, and the fixed 8-GPU DP story.
- The current stakeholder report is hand-authored against one `results.csv` under `reports/ngu800p_sweeps/`.

At the same time, [extern/modelmeter/devices/gpu.py](/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/devices/gpu.py) already exposes the shared aliases `DV100`, `DV200`, and `DV300`, along with both `p2p_bandwidth` and a dummy `cuda_tflops` for interface compatibility. That removes most of the earlier compatibility concern. The remaining work is mainly to select the device cleanly, shape metadata consistently, and fill any still-missing reporting-only fields locally when needed.

This change has an additional boundary constraint: implementation must remain inside `extern/modelmeter/models/`. Shared modules outside that subtree, including `modelmeter.devices.gpu`, are treated as read-only dependencies for this change.

The immediate stakeholder need is not just “run three more sweeps”; it is to compare NGU800P vs `DV100`/`DV200`/`DV300` on the same Wan2.1 workload and identify where bottlenecks, saturation, and 1-second SLA gaps differ.

## Goals / Non-Goals

**Goals:**
- Make the Wan2.1 DP concurrency sweep selectable by device profile rather than hard-coded to NGU800P.
- Support `DV100`, `DV200`, and `DV300` using the device data already present in `modelmeter.devices.gpu`.
- Preserve the current workload grid, DP modeling assumptions, and raw row schema as much as possible so NGU results remain comparable.
- Produce device-scoped run artifacts and generic per-device figures.
- Produce a comparison-oriented report that summarizes bottlenecks and scaling across supported hardware families.
- Keep all implementation changes within `extern/modelmeter/models/`.

**Non-Goals:**
- Changing the underlying `multi_device_cost` DP model or adding new parallelism schemas in this change.
- Calibrating device utilization against measured hardware traces.
- Expanding the Wan2.1 workload grid beyond the current supported input envelope.
- Reworking the general Wan2.1 analytic plotting stack outside the sizing/reporting path needed for this comparison.
- Editing shared ModelMeter modules outside `extern/modelmeter/models/`.

## Decisions

### 1. Generalize the existing NGU sweep script instead of adding separate DV-only scripts

The sizing logic in the current NGU script is already the canonical Wan2.1 DP sweep path. Forking it into separate `run_dv*_...` scripts would duplicate the same workload loop, model loading, stage breakdown capture, and summary generation.

Chosen approach:
- Evolve the current script into a device-parameterized sweep entrypoint.
- Accept a device selector such as `ngu800p`, `dv100`, `dv200`, or `dv300`.
- Keep the current sweep math and row construction intact unless the device abstraction requires a field normalization step.

Alternatives considered:
- Add one script per device family.
  Rejected because it would quickly drift in output schema and reporting behavior.
- Add a fully generic hardware abstraction layer across all ModelMeter models.
  Rejected for now because the immediate need is specific to Wan2.1 sizing and reporting.

### 2. Use a thin Wan2.1-local device resolver instead of a broad compatibility adapter

The Wan2.1 sweep currently assumes these fields are available on the selected device:
- tensor peak for the active precision
- `cuda_tflops`
- `io`
- `p2p_bandwidth`
- `bisection_bandwidth`

DV devices currently provide:
- `fp8_tflops`
- `fp4_tflops`
- `cuda_tflops`
- `io`
- `communication`
- `p2p_bandwidth`
- `memory_size`

Chosen approach:
- Add a small Wan2.1-local resolver/helper that:
  - maps a device name to the corresponding shared device class
  - reads the shared fields directly where they already exist
  - derives only the still-missing values needed by the current sweep/reporting flow
  - emits a consistent metadata record for `results.json`
- For DV phase 1:
  - consume shared `cuda_tflops` and `p2p_bandwidth` directly
  - derive `bisection_gb_s` locally only if reporting still expects it and the shared device does not define it
  - derive `fp16_tflops` locally only when needed for metadata or precision reporting

Rationale:
- This keeps the broader device registry untouched and avoids over-engineering a compatibility layer that the updated shared interface no longer requires.
- The current Wan2.1 DP results show `model_cuda_tflops = 0.0`, so the CUDA term is still not the active bottleneck in this analysis path today.

Alternatives considered:
- Modify the shared DV aliases in `modelmeter.devices.gpu` to mirror NGU800P exactly.
  Rejected because it violates the change boundary and hard-codes Wan2.1-specific assumptions into the shared device registry.
- Make `multi_device_cost` accept sparse device fields directly.
  Rejected because the compatibility issue is at the sweep/reporting layer, not the cost function.

### 3. Move from NGU-only output naming to device-scoped output layout

The current output tree `reports/ngu800p_sweeps/<run_id>/` and figure names like `ngu800p_full_pipeline_8gpu_throughput.svg` do not scale once multiple device families are supported.

Chosen approach:
- Store single-device runs under:
  - `extern/modelmeter/models/wan2_1/reports/hardware_sweeps/<device_name>/<run_id>/`
- Each single-device run directory contains:
  - `results.json`
  - `results.csv`
  - `summary.md`
  - `figures/`
- The per-run `figures/` directory contains the standard device-specific artifacts, for example:
  - `full_pipeline_8gpu_throughput.svg`
  - `full_pipeline_8gpu_used_rates.svg`
  - `full_pipeline_1s_sla_sizing.svg`
- Store cross-device comparison artifacts under:
  - `extern/modelmeter/models/wan2_1/reports/hardware_sweeps/comparisons/<comparison_run_id>/`
- Each comparison run directory contains:
  - `stakeholder-report.en.md`
  - `comparison-table.csv`
  - `figures/`
- The comparison `figures/` directory contains cross-device plots, for example:
  - `full_pipeline_8gpu_throughput_compare.svg`
  - `full_pipeline_8gpu_used_rates_compare.svg`
  - `full_pipeline_1s_sla_gap_compare.svg`
- Include normalized device metadata in every `results.json` payload and record the source run IDs used by any comparison report.

Directory contract:
```text
extern/modelmeter/models/wan2_1/reports/hardware_sweeps/
  ngu800p/
    <run_id>/
      results.json
      results.csv
      summary.md
      figures/
        full_pipeline_8gpu_throughput.svg
        full_pipeline_8gpu_used_rates.svg
        full_pipeline_1s_sla_sizing.svg
  dv100/
    <run_id>/
      results.json
      results.csv
      summary.md
      figures/
        full_pipeline_8gpu_throughput.svg
        full_pipeline_8gpu_used_rates.svg
        full_pipeline_1s_sla_sizing.svg
  dv200/
    <run_id>/...
  dv300/
    <run_id>/...
  comparisons/
    <comparison_run_id>/
      stakeholder-report.en.md
      comparison-table.csv
      figures/
        full_pipeline_8gpu_throughput_compare.svg
        full_pipeline_8gpu_used_rates_compare.svg
        full_pipeline_1s_sla_gap_compare.svg
```

Rationale:
- The run directory becomes self-identifying.
- Figures stay colocated with the exact run that produced them, which improves reproducibility and avoids shared-directory collisions.
- Comparison tooling can discover runs by device cleanly while keeping synthesized outputs separate from raw device sweeps.
- Existing NGU runs can coexist without migration pressure if legacy paths remain readable.

Alternatives considered:
- Keep `ngu800p_sweeps/` and add DV subfolders under it.
  Rejected because NGU would remain the misleading top-level category for a cross-device feature.

### 4. Generate a consolidated DV stakeholder report from machine-readable sweep results

The current stakeholder report is valuable, but it is effectively a frozen narrative over one NGU result set. That does not scale to four device families.

Chosen approach:
- Keep summary generation per run from `results.json`.
- Generalize the figure generator so it reads device metadata from `results.json` instead of encoding NGU labels in the script.
- Add a comparison reporting step that consumes multiple run directories and produces:
  - per-device figures using the same plot logic
  - one consolidated `stakeholder-report.en.md` for the shared “most demanding input” slice across `DV100`, `DV200`, and `DV300`

The consolidated stakeholder report should:
- use separate sections for `DV100`, `DV200`, and `DV300`
- include the same kinds of information as [stakeholder-report.en.md](/data1/huangzhe/code/llm-perf-opt/extern/modelmeter/models/wan2_1/reports/ngu800p_sweeps/stakeholder-report.en.md)
- identify which bottleneck dominates per device
- show where throughput saturates with DP batch growth
- quantify how far each device is from the 1-second SLA requirement

### Consolidated stakeholder report template

The final DV stakeholder report should be one Markdown document at:
- `extern/modelmeter/models/wan2_1/reports/hardware_sweeps/comparisons/<comparison_run_id>/stakeholder-report.en.md`

It should keep the same stakeholder-facing tone and information density as the current NGU report, but restructure the narrative so chip-independent context and model-demand metrics appear once while each DV chip gets its own device-dependent serving analysis section.

Proposed template:

```md
# DV hardware bottlenecks for Wan2.1-T2V-14B (analytic sizing)

## HEADER
- **Purpose**: Stakeholder-facing summary of predicted hardware bottlenecks when serving Wan2.1-T2V-14B on DV-class hardware under concurrent requests (DP across requests).
- **Date**: <report date>
- **Data sources (sweeps)**:
  - `models/wan2_1/reports/hardware_sweeps/dv100/<run_id>/results.csv`
  - `models/wan2_1/reports/hardware_sweeps/dv200/<run_id>/results.csv`
  - `models/wan2_1/reports/hardware_sweeps/dv300/<run_id>/results.csv`
- **Model**: ModelMeter analytic Wan2.1 full pipeline (`UMT5 text encoding + DiT diffusion + VAE decode`)
- **Precision assumptions**: `<precision summary>`
- **Parallelism modeled**: Data-parallel replication across concurrent requests (no intra-request sharding); effective GPUs = `min(batch_size, device_num)`
- **Scope note**: This is a first-order analytic model using a peak-device bottleneck timing model; treat absolute numbers as sizing guidance and validate with real profiling.

## 1) Executive summary
- One short bullet for the overall bottleneck pattern across `DV100`, `DV200`, and `DV300`.
- One short bullet for which chip is closest to the 1-second target for the most demanding supported input.
- One short bullet for how throughput scales up to `batch_size = 8` and what saturates first.
- One short bullet quantifying the 1-second SLA gap in hardware terms for each DV tier, for example required aggregate MemIO and the multiple over the 8-GPU peak.

### 1.1 At-a-glance comparison table
|Device|Primary bottleneck|Latency at batch=1 (s)|Peak throughput at 8 GPUs (videos/s)|Required MemIO for 1s (TB/s)|Gap vs 8-GPU peak|Saturation point|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|DV100|<...>|<...>|<...>|<...>|<...>|<...>|
|DV200|<...>|<...>|<...>|<...>|<...>|<...>|
|DV300|<...>|<...>|<...>|<...>|<...>|<...>|

## 2) Shared model, workload, and methodology
This section should appear once and should not be repeated inside the chip-specific sections.

### 2.1 What is Wan2.1 and what runs during inference?
- Reuse the same explanatory text and execution-flow framing as the NGU report.

### 2.2 Diffusion module architecture (high level)
- Include the same DiT architecture image used by the NGU report.
- Preserve the external source attribution for the image and model architecture reference.

### 2.3 Most demanding input structure used for comparison
- **Resolution**: <...>
- **Frames**: <...>
- **Diffusion steps**: <...>
- **Text length**: <...>

### 2.4 Modeling assumptions
- State the precision, device count range, batch sweep range, and the “effective GPUs = min(batch_size, device_num)” rule once.
- Keep the caveat that this is an analytic sizing model rather than measured serving performance.

### 2.5 Shared single-request breakdown (full pipeline)
- Include the same per-video stage totals table once because the workload demand is chip-independent for a fixed input and precision.
- Keep the explanatory note that `Diffusion (all steps)` aggregates the denoising loop across all inference steps, while `Diffusion (single step)` is shown for intuition.
- Preserve the same columns as the NGU report:
  `Stage`, `Tensor compute (TFLOPs)`, `MemIO (TB)`, `Tensor share (%)`, `MemIO share (%)`

### 2.6 Shared execution flow (single request)
- Include the same Mermaid sequence diagram once because the serving flow is the same across `DV100`, `DV200`, and `DV300`.

## 3) DV100 analysis
### 3.1 Device peaks used for hardware limit lines
- **Per GPU**: Tensor peak = <...>, MemIO peak = <...>, P2P = <...>
- **8 GPUs aggregate**: Tensor peak = <...>, MemIO peak = <...>

### 3.2 8×DV100 DP serving: batch size sweep
- Brief interpretation paragraph.
- Figure: `figures/dv100_full_pipeline_8gpu_throughput.svg`
- Figure: `figures/dv100_full_pipeline_8gpu_used_rates.svg`
- Figure: `figures/dv100_full_pipeline_1s_sla_sizing.svg`

### 3.3 Stakeholder conclusions for DV100
- Primary bottleneck.
- Scaling behavior.
- Concrete 1-second SLA gap statement in hardware terms, for example required aggregate MemIO and the multiple over the 8-GPU DV100 peak.
- Practical implication for this chip.

### Appendix A) DV100 batch sweep table
- Full table for `device_num=8` and the selected workload slice.
- Preserve the explicit filter line before the table.
- Preserve the same columns as the NGU report:
  `Batch size`, `Effective GPUs`, `Latency (s)`, `Throughput (videos/s)`, `Used Tensor (TFLOP/s)`, `Used MemIO (TB/s)`, `Compute-Max MemIO (TB/s)`, `MemIO-Max Compute (TFLOP/s)`

## 4) DV200 analysis
### 4.1 Device peaks used for hardware limit lines
- **Per GPU**: Tensor peak = <...>, MemIO peak = <...>, P2P = <...>
- **8 GPUs aggregate**: Tensor peak = <...>, MemIO peak = <...>

### 4.2 8×DV200 DP serving: batch size sweep
- Brief interpretation paragraph.
- Figure: `figures/dv200_full_pipeline_8gpu_throughput.svg`
- Figure: `figures/dv200_full_pipeline_8gpu_used_rates.svg`
- Figure: `figures/dv200_full_pipeline_1s_sla_sizing.svg`

### 4.3 Stakeholder conclusions for DV200
- Primary bottleneck.
- Scaling behavior.
- Concrete 1-second SLA gap statement in hardware terms, for example required aggregate MemIO and the multiple over the 8-GPU DV200 peak.
- Practical implication for this chip.

### Appendix B) DV200 batch sweep table
- Full table for `device_num=8` and the selected workload slice.
- Preserve the explicit filter line before the table.
- Preserve the same columns as the NGU report:
  `Batch size`, `Effective GPUs`, `Latency (s)`, `Throughput (videos/s)`, `Used Tensor (TFLOP/s)`, `Used MemIO (TB/s)`, `Compute-Max MemIO (TB/s)`, `MemIO-Max Compute (TFLOP/s)`

## 5) DV300 analysis
### 5.1 Device peaks used for hardware limit lines
- **Per GPU**: Tensor peak = <...>, MemIO peak = <...>, P2P = <...>
- **8 GPUs aggregate**: Tensor peak = <...>, MemIO peak = <...>

### 5.2 8×DV300 DP serving: batch size sweep
- Brief interpretation paragraph.
- Figure: `figures/dv300_full_pipeline_8gpu_throughput.svg`
- Figure: `figures/dv300_full_pipeline_8gpu_used_rates.svg`
- Figure: `figures/dv300_full_pipeline_1s_sla_sizing.svg`

### 5.3 Stakeholder conclusions for DV300
- Primary bottleneck.
- Scaling behavior.
- Concrete 1-second SLA gap statement in hardware terms, for example required aggregate MemIO and the multiple over the 8-GPU DV300 peak.
- Practical implication for this chip.

### Appendix C) DV300 batch sweep table
- Full table for `device_num=8` and the selected workload slice.
- Preserve the explicit filter line before the table.
- Preserve the same columns as the NGU report:
  `Batch size`, `Effective GPUs`, `Latency (s)`, `Throughput (videos/s)`, `Used Tensor (TFLOP/s)`, `Used MemIO (TB/s)`, `Compute-Max MemIO (TB/s)`, `MemIO-Max Compute (TFLOP/s)`

## 6) Cross-device conclusions
- Which chip remains MemIO-bound and which one moves closest to compute or mixed limits.
- How much each chip improves over the previous DV tier.
- Which chip is the most practical candidate for the target SLA under the current analytic assumptions.

### 6.1 Cross-device comparison figures
- Figure: `figures/full_pipeline_8gpu_throughput_compare.svg`
- Figure: `figures/full_pipeline_8gpu_used_rates_compare.svg`
- Figure: `figures/full_pipeline_1s_sla_gap_compare.svg`
```

Template rationale:
- Shared sections remove duplicated Wan2.1 explanation text, architecture references, execution flow, and chip-independent stage-demand metrics that would otherwise appear three times.
- Per-device sections remain parallel on the device-dependent parts of the current NGU report, so each chip can be read independently without repeating the same model description.
- The opening comparison table and closing cross-device conclusions let stakeholders scan the whole DV family quickly without digging through all appendices.

Alternatives considered:
- Write separate hand-authored stakeholder reports per device.
  Rejected because consistency would be difficult to maintain and stakeholders specifically want one report that covers the whole DV family.

### 5. Preserve workload and DP semantics for the first implementation pass

To make NGU and DV results comparable, the new sweep must keep the current fixed assumptions unless the user explicitly broadens scope later:
- same precision flag behavior
- same two workload points
- same `batch_size=1..32`
- same `device_num=1..8`
- same DP interpretation where effective GPUs are `min(batch_size, device_num)`

Rationale:
- This minimizes the amount of “what changed?” when comparing old NGU outputs to new generalized outputs.
- It isolates the change to hardware selection and reporting, rather than mixing in new workload or modeling dimensions.

## Risks / Trade-offs

- [DV interconnect semantics are still only partially specified] → Phase 1 consumes shared `p2p_bandwidth` directly and derives only any still-missing `bisection_bandwidth` locally, with the derived value recorded explicitly in metadata and documentation.
- [Read-only dependency on `modelmeter.devices.gpu`] → Keep the Wan2.1-local resolver limited to device selection and minimal metadata shaping so the change does not depend on registry refactors outside `extern/modelmeter/models/`.
- [Generic output relocation may break references to legacy NGU paths] → Keep legacy NGU artifacts untouched and document the new directory contract in the Wan2.1 reports README.
- [Generic figure code may still encode NGU-specific assumptions like 8-GPU saturation annotations] → Parameterize titles, filenames, and saturation markers from result metadata instead of constants.
- [Cross-device reporting can drift if row schemas diverge] → Preserve the current row schema and only add device-identifying metadata/fields rather than changing the existing metrics.
- [Stakeholders may over-interpret analytic results as measured performance] → Keep the current “first-order analytic model” caveat in generated summaries and the consolidated stakeholder report.

## Migration Plan

1. Add the Wan2.1-local device resolver and device selector.
2. Refactor the current NGU sweep script into a device-parameterized sweep without changing the underlying workload grid.
3. Re-run NGU through the generalized path and confirm the resulting curves and summary are consistent with the current NGU baseline.
4. Add generic figure generation and device-scoped output paths.
5. Add `DV100`/`DV200`/`DV300` runs and generate the first consolidated DV stakeholder report.
6. Update the Wan2.1 reports README to document the new workflow and artifact locations.

Rollback strategy:
- The original NGU run artifacts remain valid and untouched.
- If generalized reporting proves unstable, the legacy NGU stakeholder report can continue to serve as the reference output while the new DV consolidated-report path is corrected.

## Open Questions

- Do we want one “latest comparison” report path plus timestamped runs, or only timestamped comparison artifacts?
- Should DV `fp4_tflops` remain unused for this change, or do we want the device resolver to acknowledge future FP4 sweeps explicitly in metadata?
