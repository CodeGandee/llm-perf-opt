# DeepSeek-OCR Analytical Performance Report

This report summarizes analytical performance validation and scaling behavior for the DeepSeek-OCR model, using the ModelMeter analytic implementation in `extern/modelmeter/models/deepseek_ocr` and the verification and sweep scripts under `extern/modelmeter/models/deepseek_ocr/scripts`.

## Overview and Scope

This document focuses on how well the analytic FLOP/IO/memory model matches the vendor implementation and how analytic costs scale as model and workload parameters change.
Architecture and operator-level details live in `reports/deepseek-ocr-analytical-arch-report.md`; here we concentrate on:
- Per-layer sanity checks (layer-wise verification against reference implementations).
- End-to-end pipeline verification (vision-only and full model).
- Model scaling sweeps (how analytic FLOPs/IO vary with input and model parameters).

All experiments are expected to be run from the `llm-perf-opt` project root using the Pixi RTX 5090 environment unless otherwise noted.

## Per-Layer Sanity Check (Layer-wise Verification)

This section reports how analytic FLOPs and IO compare to reference implementations at the layer level.
The primary entrypoints are the `run_verify_<xxx>.py` scripts under `extern/modelmeter/models/deepseek_ocr/scripts/verify`.

### Vision Layers

This subsection covers per-layer verification for the DeepSeek-OCR vision stack (SAM encoder, CLIP/NoTP, projector).
Typical entrypoint:
- `python -m modelmeter.models.deepseek_ocr.scripts.verify.run_verify_vision`

Expected content for this subsection:
- Table of key vision layers (`PatchEmbed`, `ImageEncoderViT`, `VitModel`, `MlpProjector`, etc.) with:
  - Analytic FLOPs vs measured FLOPs (or other FLOP counters).
  - Relative differences and configured tolerances.
- Notes on any known discrepancies or approximations (for example, ignored CUDA-core-only work, FlashAttention vs SDPA differences).

### Decoder and LLaMA Layers

This subsection covers per-layer verification for the DeepSeek-V2 decoder stack and LLaMA primitives.
Typical entrypoints:
- `python -m modelmeter.models.deepseek_ocr.scripts.verify.run_verify_decoder`
- `python -m modelmeter.models.deepseek_ocr.scripts.verify.run_verify_llama`

Expected content:
- Tables or bullet summaries for:
  - `DeepseekV2MLP`, `DeepseekV2MoE`, `MoEGate`, `DeepseekV2RMSNorm`, `DeepseekV2DecoderLayer`.
  - `LlamaFlashAttention2`, `LlamaRotaryEmbedding` and any other LLaMA primitives.
- Relative error ranges and whether they fall within configured tolerances for each operator family.

### Core Aggregation and Prefill/Decode Consistency

This subsection checks that analytic aggregation across layers is self-consistent and matches reference models at the decoder level.
Typical entrypoints:
- `python -m modelmeter.models.deepseek_ocr.scripts.verify.run_verify_core`
- `python -m modelmeter.models.deepseek_ocr.scripts.verify.run_verify_prefill_decode`

Expected content:
- Verification that `DeepseekOCRModel` FLOPs equal the sum of its configured sublayers (vision + decoder + head) within floating-point precision.
- Decoder-only prefill vs decode FLOP comparisons versus reference `DeepseekV2ForCausalLM` or similar, including any known decode-specific gaps.

## End-to-End Analytical vs Vendor FLOPs

This section reports end-to-end FLOP alignment between the analytic model and vendor implementation across realistic OCR workloads.
It builds on the per-layer checks to validate the full pipeline under representative inputs.

### Vision-Only Pipelines

This subsection covers vision-only verification, both in no-crop (global view) and crop-mode (global + dynamic crops) scenarios.
Typical entrypoints:
- `python -m modelmeter.models.deepseek_ocr.scripts.verify.run_verify_end2end_vision_nocrop`
- `python -m modelmeter.models.deepseek_ocr.scripts.verify.run_verify_end2end_vision`

Expected content:
- For each scenario:
  - Analytic vision FLOPs vs vendor FLOPs for a canonical document image.
  - Relative differences and comments on shape modeling (multi-view layout, token counts).
- Discussion of any remaining vision-specific gaps and how they relate to shape modeling assumptions.

### Full Model Prefill + Decode

This subsection validates the full DeepSeek-OCR pipeline (vision + decoder + head) for realistic OCR workloads, including both prefill and bounded decode.
Typical entrypoints:
- `python -m modelmeter.models.deepseek_ocr.scripts.verify.run_verify_end2end_prefill_decode`
- `python -m modelmeter.models.deepseek_ocr.scripts.verify.run_verify_end2end`

Expected content:
- Separate summaries for:
  - Prefill-only FLOPs (vision + decoder + head) vs vendor reference.
  - Decode-only FLOPs vs vendor reference (including any known decode mismatches).
  - Total end-to-end FLOPs vs vendor `generate`.
- Interpretation of how well the analytic model brackets vendor compute at the pipeline level and where deviations remain.

## Model Scaling and Workload Sweeps

This section explores how analytic FLOPs, IO, and memory usage scale as model and input parameters change.
It relies on sweep scripts under `extern/modelmeter/models/deepseek_ocr/scripts/sweep` and any associated helpers.

### Vision Input Shape Sweeps

This subsection studies how vision compute changes with input resolution and crop configuration.
Typical entrypoint:
- `python -m modelmeter.models.deepseek_ocr.scripts.sweep.sweep-vision-input-shape`

Expected content:
- FLOP vs resolution plots or tables for:
  - No-crop global views (varying base size).
  - Crop-mode configurations (varying image size and crop grids).
- Discussion of regimes where vision cost is dominated by SAM vs CLIP vs projector and how that relates to document layout.

### Sequence Length and Decoder Sweeps

This subsection focuses on how decoder FLOPs and KV-cache memory scale with:
- Prefill context length (`S_prefill`).
- Decode length (`K` tokens).
- Batch size (`B`) and head configuration.

Expected content:
- Analytic curves for:
  - `F_prefill_total` and `F_decode_total` vs `S_prefill`, `K`, and `B`.
  - KV-cache memory vs `S_prefill + K`.
- Identification of regimes where prefill dominates vs decode, and where KV memory becomes a primary constraint.

### Combined Workload Profiles

This subsection looks at realistic workload profiles that combine image resolution, context length, and decode length (for example, different OCR workload IDs).

Expected content:
- Summary tables for a few canonical workload profiles (e.g., `<WORKLOAD_PROFILE_ID>` values):
  - Vision FLOPs.
  - Prefill FLOPs.
  - Decode FLOPs (per token and total).
- High-level comments on how these analytic workloads map to expected runtime and MFU when combined with hardware peak tables (referencing `docs/analyzer-mfu.md` rather than reproducing MFU analysis here).
