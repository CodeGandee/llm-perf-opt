# Data Model: DeepSeek‑OCR Analytic Modeling in ModelMeter

This document defines domain entities, fields, and relationships for analytic modeling of DeepSeek‑OCR. Internal models
should be implemented with `attrs` (for data‑only containers) and live alongside existing Stage 1 profiling models in
`src/llm_perf_opt/data/models.py` or a nearby module. Web/contract schemas are described in
`specs/001-deepseek-ocr-modelmeter/contracts/python-contracts.md`.

## Entities

1) DeepSeekOCRModelSpec
- model_id: string — Canonical identifier for the model (e.g., `"deepseek-ai/DeepSeek-OCR"`).
- model_variant: string — Internal variant name (e.g., `"deepseek-ocr-v1-base"`).
- hf_revision: string | null — Optional Hugging Face revision or git SHA used for the checkpoint.
- config_path: string — Absolute path to the local config/model definition used for analysis.
- hidden_size: int — Transformer hidden dimension for the decoder.
- intermediate_size: int — MLP inner dimension for the decoder.
- num_layers: int — Number of decoder layers.
- num_attention_heads: int — Number of attention heads in the decoder.
- vision_backbone: enum {sam_vit_b, clip_vit_l, other} — Vision encoder backbone family.
- uses_moe: bool — Whether the decoder stack uses MoE blocks.
- notes: string — Free‑form notes (e.g., attention implementation, rotary variant).

2) OCRWorkloadProfile
- profile_id: string — Workload identifier (e.g., `"dsocr-standard-v1"`).
- description: string — Human‑readable description of the workload.
- seq_len: int — Representative total sequence length (tokens, including image tokens).
- base_size: int — Global view padding size (pixels, e.g., 1024).
- image_size: int — Local crop size (pixels, e.g., 640).
- crop_mode: bool — Whether dynamic cropping is enabled.
- max_new_tokens: int — Maximum generated tokens used for calibration.
- doc_kind: enum {text_heavy, mixed_layout, image_rich, synthetic} — Logical document category.
- num_pages: int — Representative number of pages (1 for this feature).

3) AnalyticModuleNode
- module_id: string — Stable identifier (e.g., `"vision/image_encoder_vit"`, `"decoder/block[0-23]"`).
- name: string — Human‑readable module name (e.g., `"ImageEncoderViT"`, `"DeepseekV2DecoderLayer[0-23]"`).
- qualified_class_name: string — Fully qualified Python class name.
- stage: enum {vision, projector, prefill, decode, other} — Stage attribution.
- parent_id: string | null — Parent module id; null for the root.
- children: list[string] — Child module ids.
- repetition: enum {none, for, parfor} — Repetition semantics for grouped modules.
- repetition_count: int | null — Count `N` used with `for`/`parfor` semantics.
- constructor_params: map[string, int | float | string | bool] — Shape‑relevant constructor/config parameters.

4) OperatorCategory
- category_id: string — Identifier (e.g., `"conv2d"`, `"linear"`, `"norm"`, `"attention"`, `"activation"`).
- display_name: string — Human‑readable label.
- description: string — Short description of the operator family.
- match_classes: list[string] — List of class names or qualified names that map into this category.

5) ModuleMetricsSnapshot
- module_id: string — Reference to AnalyticModuleNode.module_id.
- profile_id: string — Reference to OCRWorkloadProfile.profile_id.
- calls: int — Total number of calls under the workload.
- total_time_ms: float — Predicted total runtime for this module (forward only).
- total_flops_tflops: float — Predicted FLOPs for this module (TFLOPs, forward).
- total_io_tb: float — Predicted I/O traffic (Tb, forward).
- memory_weights_gb: float — Weight memory footprint (GB).
- memory_activations_gb: float — Activation memory footprint (GB).
- memory_kvcache_gb: float — KV‑cache memory footprint (GB), where applicable.
- share_of_model_time: float — Fraction of model total runtime attributed to this module (0..1+).
- operator_breakdown: list[OperatorMetrics] — Operator‑level breakdown for this module.

6) OperatorMetrics
- category_id: string — Reference to OperatorCategory.category_id.
- calls: int — Total operator calls contributing to the module.
- flops_tflops: float — FLOPs attributed to this operator category within the module.
- io_tb: float — I/O traffic attributed to this operator category.
- share_of_module_flops: float — Fraction of module FLOPs for this category (0..1+).

7) AnalyticModelReport
- report_id: string — Unique identifier for this analytic report.
- model: DeepSeekOCRModelSpec — Model specification used for the analysis.
- workload: OCRWorkloadProfile — Workload profile for which the analytic model is valid.
- modules: list[AnalyticModuleNode] — Module hierarchy definition.
- operator_categories: list[OperatorCategory] — Operator category definitions.
- module_metrics: list[ModuleMetricsSnapshot] — Per-module metrics under the workload.
- profile_run_id: string | null — Optional link to a Stage 1/Stage 2 profiling run used for later validation (not required in this development stage).
- predicted_total_time_ms: float — Sum of analytically predicted module runtimes.
- measured_total_time_ms: float | null — Optional measured runtime for future cross-check; may be omitted in this development stage.
- predicted_vs_measured_ratio: float | null — Predicted / measured ratio (or null if no measurement); reserved for future runtime comparison.
- notes: string — Stakeholder-oriented notes and caveats.
- layer_docs_dir: string — Absolute path to the directory containing generated Markdown documentation for analytic
  layers/operators (for example, per-layer `.md` files explaining definitions, formulas, and assumptions).

8) TargetOperatorList
- snapshot_id: string — Identifier for the static operator snapshot (e.g., TorchInfo timestamp).
- source_artifact_dir: string — Absolute path to the directory with TorchInfo artifacts.
- layers_md_path: string — Absolute path to `torchinfo-unique-layers.md`.
- layers_json_path: string — Absolute path to `torchinfo-unique-layers.json`.
- stages_json_path: string — Absolute path to `torchinfo-stages.json`.
- operators: list[OperatorSpec] — Flattened operator list as parsed from TorchInfo.

9) OperatorSpec
- class_name: string — Operator or module class name.
- class_name_qualified: string — Fully qualified class name from TorchInfo.
- is_pytorch_builtin: bool — True if the operator lives under `torch.nn`.
- is_custom: bool — True if the operator is vendor/custom code.
- children_classes: list[string] — Direct children class names (as reported by TorchInfo).
- default_category_id: string — Suggested OperatorCategory for this operator.

## Relationships

- AnalyticModelReport references exactly one DeepSeekOCRModelSpec and one OCRWorkloadProfile.
- AnalyticModelReport.modules forms a rooted tree via `parent_id`/`children`.
- Each ModuleMetricsSnapshot.module_id must match one AnalyticModuleNode.module_id.
- ModuleMetricsSnapshot.profile_id must match OCRWorkloadProfile.profile_id.
- OperatorMetrics.category_id must match OperatorCategory.category_id.
- TargetOperatorList.operators provides the authoritative mapping from vendor classes to analytic categories.
- AnalyticModelReport.profile_run_id, when present, should match a Stage 1/Stage 2 profiling run id
  (e.g., `LLMProfileReport.run_id`) to support future validation against measured metrics (not required in this stage).

## Validation Rules

- All absolute path fields (e.g., `config_path`, `source_artifact_dir`, `*_path`, `layer_docs_dir`) MUST be absolute paths.
- Numeric metrics (FLOPs, I/O, memory, time, counts) MUST be non‑negative.
- `share_of_model_time`, `share_of_module_flops` SHOULD be in [0, 1.5]; values >1 indicate estimator drift and should
  be flagged.
- For this development stage, `predicted_total_time_ms` and related ratios are interpreted as theoretical estimates only.
  Automated numeric validation against measured runtimes is not required; instead, experts will review both the analytic
  report (JSON/YAML) and the generated Markdown documentation for internal consistency and plausibility.
- `modules` MUST form an acyclic graph (tree or forest with a designated root).
- `module_id`, `category_id`, and `profile_id` MUST be unique within a report.

## Contracts Alignment

Contracts in `/workspace/code/llm-perf-opt/specs/001-deepseek-ocr-modelmeter/contracts/` define request/response shapes
using the following names (see `python-contracts.md`):

- DeepSeekOCRAnalyticRequest
- DeepSeekOCRAnalyticAccepted
- DeepSeekOCRAnalyticReportSummary

Domain models above intentionally align with these contracts:

- DeepSeekOCRAnalyticReportSummary will expose a subset of AnalyticModelReport fields, focused on:
  - `report_id`, `model_variant`, `workload_profile_id`
  - top‑N modules by time/FLOPs
  - high‑level theoretical runtime summaries suitable for human review (runtime comparison fields are optional and may remain unset in this stage).
- DeepSeekOCRAnalyticRequest references `model_id`, `model_variant`, `workload_profile_id`, and optionally an existing
  profiling run id.
- TargetOperatorList and OperatorCategory remain internal; downstream tools consume only the flattened analytic model
  plus module metrics.

Conversion between domain and contracts SHOULD be implemented via `cattrs` to keep serialization logic centralized and
type‑safe.
