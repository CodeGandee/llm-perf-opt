# Data Model: Wan2.1 Analytic FLOP Report

This feature introduces a Wan2.1-specific model spec and workload profile while reusing shared analytic report schema components (module tree + operator categories + metrics snapshots) across DeepSeek-OCR and Wan2.1.

## Entities

- Wan2_1ModelSpec (new)
  - model_id: string (canonical id, e.g., `wan2.1-t2v-14b`)
  - model_variant: string (variant tag, e.g., `t2v-14b`)
  - config_path: absolute path to the model metadata source used for analytic parameters (typically under `models/wan2.1-t2v-14b/source-data/`)
  - transformer:
    - hidden_size: int
    - num_layers: int
    - num_attention_heads: int
    - head_dim: int
    - mlp_intermediate_size: int
  - tokenization / latent geometry:
    - vae_downsample_factor: int (spatial downsample ratio from pixels to latents)
    - patch_size: int (patching factor applied to latents before transformer)
    - latent_channels: int
  - notes: string (free-form disclosure of defaults and known omissions)

- Wan2_1WorkloadProfile (new)
  - profile_id: string
  - description: string
  - batch_size: int
  - num_frames: int
  - height: int
  - width: int
  - num_inference_steps: int
  - text_len: int

- AnalyticModuleNode (shared)
  - Purpose: stable hierarchical identifier and display metadata for each analytic layer/module.
  - Notes: Wan2.1 reports must use deterministic `module_id` values for transformer blocks and key subcomponents to enable layer-by-layer comparison.

- OperatorCategory (shared)
  - Purpose: consistent grouping labels for per-module operator breakdowns (for example, attention projections, attention core, MLP projections, normalization).

- ModuleMetricsSnapshot (shared)
  - Purpose: the numeric metrics for a module under a workload (calls, total FLOPs, I/O, memory, share of total).

- Wan2_1AnalyticModelReport (new, model-specific wrapper around shared schema)
  - report_id: string
  - model: Wan2_1ModelSpec
  - workload: Wan2_1WorkloadProfile
  - modules: AnalyticModuleNode[]
  - operator_categories: OperatorCategory[]
  - module_metrics: ModuleMetricsSnapshot[]
  - predicted_total_time_ms: float (optional/approximate; may be 0 when only FLOPs are modeled)
  - notes: string
  - layer_docs_dir: absolute path to per-layer markdown docs (optional; consistent with existing static-analysis conventions)

## Relationships

- Wan2_1AnalyticModelReport.model is 1:1 with Wan2_1ModelSpec.
- Wan2_1AnalyticModelReport.workload is 1:1 with Wan2_1WorkloadProfile.
- Wan2_1AnalyticModelReport.modules forms a tree through `parent_id` and `children` references; each ModuleMetricsSnapshot references exactly one AnalyticModuleNode by `module_id`.

## Validation Rules

- All filesystem paths stored in reports MUST be absolute paths.
- All integer workload parameters MUST be positive; `num_inference_steps` MUST be ≥ 1.
- All FLOP / I/O / memory metrics MUST be non-negative and finite.
- End-to-end totals MUST equal the sum of per-module values (within floating point tolerance) for each metric that is reported.
- Wan2.1 diffusion transformer modules MUST report KV-cache memory as 0 unless explicitly justified in the report notes/metadata.

## State Transitions

1. (Inputs) model metadata + workload profile → (instantiate) ModelMeter analytic layer tree → (evaluate) per-module costs → (assemble) Wan2_1AnalyticModelReport → (emit) `report.json` + optional `summary.md` under `tmp/profile-output/<run_id>/static_analysis/wan2_1/`.
