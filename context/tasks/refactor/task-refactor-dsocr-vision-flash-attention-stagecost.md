Refactor Plan: DeepSeek-OCR Vision Flash-Attention StageCost Wiring
===================================================================

What to Refactor
----------------

- Vision analytic configuration and composites:
  - `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`
  - `extern/modelmeter/models/deepseek_ocr/configs/model/deepseek_ocr_root.default.yaml`
- Vision analytic layers used in the CLIP stack:
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/notp_attention.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/notp_transformer_block.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/notp_transformer.py`
- Vision sweep scripts and plotting helpers:
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-vision-crops.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-vision-prefill.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/_e2e_plot_utils.py`
- (Optional sanity hooks) Vision verification helpers:
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_vision.py`

The goal is to ensure that:

- Analytic **normal-attention** and **flash-attention** modes for the vision stack are wired as truly distinct configurations.
- StageCost outputs (including `io_tb`, `activations_gb`, `flops_tensor_tflops`, `flops_cuda_tflops`) reflect the expected FlashAttention behavior, and sweeps/plots show meaningful differences where they should.

Why Refactor
------------

- **Observed issue (current state)**
  - In the vision sweeps, especially `sweep-vision-crops.py`, the analytic StageCost entries for the two purported vision modes:
    - `vision` (normal attention) and
    - `vision_flash` (flash attention)
    are **numerically identical** for every point in the sweep:
    - `flops_tflops`, `io_tb`, `activations_gb`, `flops_tensor_tflops`, and `flops_cuda_tflops` all match bit-for-bit.
  - As a result:
    - `stagecost_arithmetic_intensity.svg` and other StageCost plots show overlapping curves for analytic normal vs flash attention.
    - The FLOP-split figures (`vision_flops_split_*`) show a single effective analytic curve, even though they are labeled as flash-attention-only.
  - This happens despite the fact that:
    - The per-layer analytic vision attention implementations (`Attention` and `NoTPAttention`) do branch on `use_flash_attention` and, in isolation, produce different `io_tb` / `activations_gb` for flash vs non-flash settings.
    - Decoder sweeps already show clear StageCost differences between normal and flash attention for the decoder stack.

- **Root cause (analytic wiring vs layer math)**
  - The analytic layers are capable of modeling flash-specific behavior, but the way Hydra configs and composites are wired means:
    - Both `vision` and `vision_flash` composites used in sweeps are effectively bound to the *same* underlying CLIP NoTP transformer configuration (same `use_flash_attention`), or the override is applied to a config path that is not actually used by the instantiated composite.
    - The sweep scripts attempt to toggle `use_flash_attention` by mutating `cfg_crops.vision.notp_attention.use_flash_attention`, but this does not change the effective `use_flash_attention` of the layers that contribute to the StageCost for the composite.
  - In practice, this yields two analytic models with identical StageCost behavior under different labels, rather than truly distinct normal vs flash attention modes.

- **Why we need to fix this**
  - We rely on the analytic model, not the vendor model, to answer “what-if” questions about:
    - How FlashAttention changes **activation memory and I/O** in the vision stack.
    - How Tensor Core vs CUDA-core FLOPs break down in the vision pipeline.
    - How much TFLOPs/s and bandwidth are required to hit given TTFT targets under different attention implementations.
  - If normal and flash analytic modes are not actually distinct at the pipeline level:
    - Vision sweeps and responsive reports are effectively calibrated to a single configuration, even though plots and JSON schemas imply otherwise.
    - Capacity planning and optimization discussions around vision FlashAttention are based on degenerate or misleading data.

Without a clean, config-level separation of analytic normal vs flash modes in the vision pipeline—and sweeps that exercise those modes explicitly—downstream tools (sweeps, responsive reporting, documentation) cannot correctly quantify the impact of FlashAttention in the vision stack.

TODO
----

- [ ] Add explicit normal and flash NoTP attention variants (including blocks, transformers, and ViT wrappers) in `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`, keeping `vision` as a backward-compatible alias.
- [ ] Expose `vision_layer_normal` and `vision_layer_flash` (plus a default `vision_layer` alias) in `extern/modelmeter/models/deepseek_ocr/configs/model/deepseek_ocr_root.default.yaml`, and ensure `DeepseekOCRModel.from_layers` can accept any extra Hydra parameters without failing.
- [ ] Refactor `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-vision-crops.py` to build separate analytic configs for normal and flash attention by selecting `vision.vision_normal` and `vision.vision_flash` composites instead of mutating `vision.notp_attention.use_flash_attention`.
- [ ] Refactor `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-vision-prefill.py` to follow the same pattern so StageCost prefill sweeps exercise distinct analytic normal vs flash vision stacks.
- [ ] Update vision shape overrides in `extern/modelmeter/models/deepseek_ocr/layers/core/vision_shape_config.py` so both normal and flash NoTP variants receive consistent `batch_size` and `seq_len` settings for global and crop branches.
- [ ] Add a small regression test that builds analytic `vision_normal` and `vision_flash` models for a representative workload and asserts that `flops_tflops` match while `io_tb` or `activations_gb` differ.
- [ ] Sanity-check that decoder and SAM attention paths keep their existing flash-attention wiring and remain consistent with the new vision normal/flash analytic variants.

Implementation Summary
----------------------

- Implemented explicit `notp_attention_normal` / `notp_attention_flash` variants and corresponding `notp_block_*`, `notp_transformer_*`, `vit_model_*`, and composite `vision_*` stacks in `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`, keeping `vision` and `vision_nocrop` as backward-compatible aliases that default to flash attention.
- Exposed `vision_layer_normal`, `vision_layer_flash`, `vision_layer_nocrop_normal`, and `vision_layer_nocrop_flash` (plus `vision_layer` and `vision_layer_nocrop` aliases) in `extern/modelmeter/models/deepseek_ocr/configs/model/deepseek_ocr_root.default.yaml` so Hydra callers can select normal vs flash analytic vision stacks explicitly.
- Updated `DeepseekOCRModel.from_layers` in `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py` to accept and ignore extra keyword arguments, making it resilient to future Hydra config extensions around vision-layer selection.
- Refactored `extern/modelmeter/models/deepseek_ocr/layers/core/vision_shape_config.py` so `apply_vision_overrides_for_workload` configures both normal and flash NoTP variants (global and crops) with consistent `batch_size` and `seq_len`, and keeps `vit_model` / `vit_model_crops` aliases bound to the flash variants for backwards compatibility.
- Updated `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-vision-crops.py` to select `model.vision_layer` as `vision.vision_normal` vs `vision.vision_flash` instead of mutating `vision.notp_attention.use_flash_attention` in-place, ensuring the full vision stack is switched between normal and flash modes coherently.
- Extended `extern/modelmeter/models/deepseek_ocr/scripts/verify/analytic_modes.py` so `build_analytic_model_for_mode` binds `model.vision_layer` to either `vision_layer_normal` or `vision_layer_flash` when present, aligning vision and decoder attention modes for prefill sweeps such as `sweep-e2e-vision-prefill.py`.
- Added regression tests in `tests/unit/deepseek_ocr/test_vision_flash_attention_variants.py` that confirm `NoTPAttention` flash vs normal variants share the same FLOPs but differ in I/O and activation memory, and that analytic `vision_normal` vs `vision_flash` composites yield identical `flops_tflops` but lower `io_tb` and `activations_gb` for the flash-attention path.

How to Refactor
---------------

1. Introduce explicit analytic vision variants in Hydra config
   - Extend `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml` with two explicit NoTP attention variants:
     - `notp_attention_normal` – `use_flash_attention: false`
     - `notp_attention_flash` – `use_flash_attention: true`
   - Add corresponding block/transformer variants:
     - `notp_block_normal` / `notp_block_flash`
     - `notp_transformer_normal` / `notp_transformer_flash`
   - Define two ViT composites:
     - `vit_model_normal` (using `notp_transformer_normal`)
     - `vit_model_flash` (using `notp_transformer_flash`)
   - Define two top-level analytic vision composites:
     - `vision_flash` that uses `vit_model_flash` (and, if desired, flash-attention settings for the SAM encoder too).
     - `vision_normal` that uses `vit_model_normal`.

   **Before (single CLIP stack):**

   ```yaml
   notp_attention:
     _target_: modelmeter.models.deepseek_ocr.layers.vision.notp_attention.NoTPAttention
     hidden_size: ${hf.vision.clip_hidden_size}
     num_heads: 16
     seq_len: 257
     batch_size: ${runtime.batch_size}
     use_flash_attention: true

   notp_block:
     _target_: modelmeter.models.deepseek_ocr.layers.vision.notp_transformer_block.NoTPTransformerBlock
     attention: ${vision.notp_attention}
     mlp: ${vision.notp_feedforward}
   ```

   **After (normal + flash variants):**

   ```yaml
   notp_attention_normal:
     _target_: modelmeter.models.deepseek_ocr.layers.vision.notp_attention.NoTPAttention
     hidden_size: ${hf.vision.clip_hidden_size}
     num_heads: 16
     seq_len: 257
     batch_size: ${runtime.batch_size}
     use_flash_attention: false

   notp_attention_flash:
     _target_: modelmeter.models.deepseek_ocr.layers.vision.notp_attention.NoTPAttention
     hidden_size: ${hf.vision.clip_hidden_size}
     num_heads: 16
     seq_len: 257
     batch_size: ${runtime.batch_size}
     use_flash_attention: true

   notp_block_normal:
     _target_: modelmeter.models.deepseek_ocr.layers.vision.notp_transformer_block.NoTPTransformerBlock
     attention: ${vision.notp_attention_normal}
     mlp: ${vision.notp_feedforward}

   notp_block_flash:
     _target_: modelmeter.models.deepseek_ocr.layers.vision.notp_transformer_block.NoTPTransformerBlock
     attention: ${vision.notp_attention_flash}
     mlp: ${vision.notp_feedforward}
   ```

   (The same duplication pattern applies to `notp_transformer` and `vit_model`.)

2. Make vision composites selectable at the root model level
   - Update `extern/modelmeter/models/deepseek_ocr/configs/model/deepseek_ocr_root.default.yaml` to expose both `vision_layer_normal` and `vision_layer_flash`, for example:

   ```yaml
   # Existing
   vision_layer: ${vision.vision}

   # New
   vision_layer_normal: ${vision.vision_normal}
   vision_layer_flash: ${vision.vision_flash}

   # Optional: default which is used when not explicitly overridden.
   vision_layer: ${vision.vision_flash}
   ```

   - Ensure `DeepseekOCRModel.from_layers` can pick which vision composite to attach based on a Hydra parameter (for example, `model.vision_variant` or by binding `vision_layer` directly):
     - Normal analytic vision: binds `vision_layer_normal`.
     - Flash analytic vision: binds `vision_layer_flash`.

3. Refactor vision sweeps to use explicit analytic variants instead of ad-hoc overrides
   - In `sweep-vision-crops.py`:
     - Stop toggling `cfg_crops.vision.notp_attention.use_flash_attention` directly.
     - Instead, construct the two analytic configs by selecting different vision composites:
       - `cfg_crops_normal` with `model.vision_layer = ${vision.vision_normal}`.
       - `cfg_crops_flash` with `model.vision_layer = ${vision.vision_flash}`.
     - Instantiate:
       - `model_crops_normal = build_analytic_model(cfg_crops_normal)`
       - `model_crops_flash = build_analytic_model(cfg_crops_flash)`
     - Preserve the `stage_costs` structure:

   **Before (current sweep snippet):**

   ```python
   cfg_crops_normal: DictConfig = copy.deepcopy(cfg_crops)
   cfg_crops_flash: DictConfig = copy.deepcopy(cfg_crops)

   try:
       cfg_crops_normal.vision.notp_attention.use_flash_attention = False
   except Exception:
       pass
   try:
       cfg_crops_flash.vision.notp_attention.use_flash_attention = True
   except Exception:
       pass

   model_crops_normal = build_analytic_model(cfg_crops_normal)
   model_crops_flash = build_analytic_model(cfg_crops_flash)
   ```

   **After (explicit composites):**

   ```python
   cfg_crops_normal: DictConfig = copy.deepcopy(cfg_crops)
   cfg_crops_flash: DictConfig = copy.deepcopy(cfg_crops)

   cfg_crops_normal.model.vision_layer = cfg_crops_normal.vision.vision_normal
   cfg_crops_flash.model.vision_layer = cfg_crops_flash.vision.vision_flash

   model_crops_normal = build_analytic_model(cfg_crops_normal)
   model_crops_flash = build_analytic_model(cfg_crops_flash)
   ```

   - Apply the same pattern in `sweep-e2e-vision-prefill.py` (for the prefill stage) if a vision-only flash vs normal comparison is desired there as well.

4. Verify StageCost differences at the layer and sweep levels
   - Add a small debug script or unit test to confirm that, for a fixed `(B, S, C)`, `NoTPAttention(use_flash_attention=True)` and `NoTPAttention(False)` produce:
     - Same `flops_tflops` but different `io_tb` and `activations_gb` (flash having lower values).
   - Re-run:
     - `sweep-vision-crops.py` with `datasets/manual/example.png`.
     - `sweep-e2e-vision-prefill.py` for a representative image.
   - Confirm:
     - `stagecost_*.svg` for vision now show distinct analytic curves for normal vs flash attention in I/O and activation metrics.
     - FLOP-only plots show only the flash-analytic curve plus vendor, as specified in `sweep-requirements.md`.
     - FLOP-split plots (`vision_flops_split_*`, `e2e_vision_prefill_flops_split_*`) show total/Tensor/CUDA FLOPs for flash attention in both linear and log scales.

5. Keep decoder and SAM paths consistent
   - Ensure that the new normal/flash wiring in the vision stack is consistent with:
     - Decoder analytic modes (`AnalyticFlopMode.FULL_NORMAL_ATTENTION`, `FULL_FLASH_ATTENTION`).
     - SAM attention’s `use_flash_attention` flag in `extern/modelmeter/models/deepseek_ocr/layers/vision/attention.py`.
   - If necessary, introduce explicit SAM attention variants (normal vs flash) and bind them into the `vision_normal` / `vision_flash` composites so the entire vision stack respects the mode.

Impact Analysis
---------------

- **Functional impact**
  - No change to the DeepSeek-OCR vendor implementation: all changes are confined to analytic configs, analytic layers, and sweep scripts.
  - Analytic FLOPs (`flops_tflops`) for vision should remain unchanged between normal and flash variants, as FlashAttention is a performance/memory optimization, not a FLOP reduction.
  - Analytic I/O and activation memory metrics (`io_tb`, `activations_gb`) for the CLIP vision stack will differ between normal and flash variants, as the S² attention-buffer term is present only in the non-flash configuration.
  - StageCost-based sweeps and plots will now show:
    - Distinct analytic normal vs flash curves for memory-related metrics.
    - A single analytic flash curve for FLOP-only plots (plus vendor).
    - Explicit Tensor vs CUDA FLOP splits for the flash-attention analytic path.

- **Risks**
  - Hydra config duplication for normal/flash variants could drift if not maintained carefully, leading to inconsistent model geometry between variants.
  - If sweeps or verification scripts rely on the old `vision.vision` composite implicitly, they may need small updates to select `vision_normal` or `vision_flash` explicitly.
  - FLOP-split plots and StageCost consumers (SQLite export, downstream notebooks) might need minor schema awareness for the new fields (`flops_tensor_tflops`, `flops_cuda_tflops`) if they start relying on them.

- **Mitigation**
  - Centralize shared parameters between `notp_attention_normal` and `notp_attention_flash` (and the corresponding blocks/transformers) to avoid copy-paste errors.
  - Keep `vision` as a backward-compatible alias (e.g. bound to `vision_flash`) so existing scripts that only refer to `vision` still work.
  - Add a small regression test that:
    - Builds `vision_normal` and `vision_flash` analytic models.
    - Asserts that `flops_tflops` match but `io_tb` or `activations_gb` differ for a representative workload.

Expected Outcome
----------------

Once this refactor is complete:

- The analytic vision stack will have **two clearly separated modes**:
  - Normal attention (eager-style SDPA) for sanity-aligning with vendor behavior.
  - Flash attention (FlashAttention-style SDPA) for modeling future/high-performance scenarios.
- Vision sweeps will show:
  - FLOP-only plots where the analytic curve corresponds to flash attention only, plus vendor normal-attention FLOPs.
  - StageCost plots where:
    - `vision` vs `vision_flash` separate clearly on `io_tb`, `activations_gb`, and `arithmetic_intensity`.
    - Tensor/Core FLOP splits are visible for the flash-attention analytic path in dedicated split plots.
- Responsive TTFT reporting and capacity planning will be able to:
  - Distinguish the required TFLOPs/s and memory bandwidth for normal vs flash analytic configurations in the vision stack.
  - Reason more accurately about how much benefit FlashAttention brings at the vision stage, not just in the decoder.

References
----------

- Analytic StageCost structures:
  - `extern/modelmeter/models/common/stage_cost.py`
  - `extern/modelmeter/models/common/__init__.py`
- Vision analytic layers:
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/attention.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/notp_attention.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/notp_transformer_block.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/vision/notp_transformer.py`
- Vision configs and composites:
  - `extern/modelmeter/models/deepseek_ocr/configs/vision/deepseek_ocr_base.yaml`
  - `extern/modelmeter/models/deepseek_ocr/configs/model/deepseek_ocr_root.default.yaml`
- Sweep scripts and helpers:
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-vision-crops.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-vision-prefill.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/_e2e_plot_utils.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-requirements.md`
- Vision verification and FLOP-counter caveats:
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_vision.py`
  - `extern/modelmeter/models/deepseek_ocr/docs/issues/issue-vision-flop-counter-mismatch.md`
  - `extern/modelmeter/models/deepseek_ocr/docs/caveats/caveats-torch-flop-counter.md`
- Third-party libraries (for context on FlashAttention and SDPA kernels):
  - PyTorch fused scaled-dot-product attention:
    - Context7 library id: `/pytorch/pytorch`
  - FlashAttention reference implementation:
    - Context7 library id: `/dao-ailab/flash-attention`
