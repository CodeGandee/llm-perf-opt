Refactor Plan: DeepSeek-OCR End-to-End FLOP Verification Modes
================================================================

What to Refactor
----------------

This refactor focuses on the end-to-end FLOP verification and sweep paths for DeepSeek-OCR, specifically:

- Verification scripts under `extern/modelmeter/models/deepseek_ocr/scripts/verify/`:
  - `run_verify_end2end.py`
  - `run_verify_end2end_prefill_decode.py`
  - `run_verify_prefill_decode.py`
  - `run_verify_core.py`
- E2E sweep scripts that rely on the same vendor/analytic wiring:
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-crops.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-vision-prefill.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-decode.py`
- The way analytic FLOP modes are controlled:
  - `DeepseekOCRModel._set_ignore_torch_unsupported_flop_count(...)`
  - Decoder attention implementation (eager vs FlashAttention2) for the analytic model only.

We are **not** changing the existing per-layer `.verify_by_impl()` helpers; this plan is about pipeline- and stage-level FLOP verification and visualization.


Why Refactor
------------

Through recent debugging we learned several important facts:

1. The DeepSeek-OCR vendor model used in our verification scripts currently runs with **eager** `LlamaAttention` (not `LlamaFlashAttention2`), so:
   - `torch.utils.flop_counter.FlopCounterMode` sees the attention-core FLOPs as standard matmul/bmm ops.
   - Full analytic decode FLOPs (with `ignore_torch_unsupported_flop_count=False`) match the vendor FLOPs better than the “torch-visible” variant.
2. If we switch to `LlamaFlashAttention2`, the flop counter **misses almost exactly the fused FlashAttention2 core** FLOPs:
   - Projections (Q/K/V/O) are still counted.
   - The core FlashAttention2 math (QKᵀ + softmax·V) is effectively invisible to `FlopCounterMode`, as shown by `verify_flash_attention_flops.py` and `verify_llama_attention.py`.
3. We want to support **two analytic modes**:
   - A “vendor-aligned” analytic mode that matches what the torch flop counter sees.
   - A “true” analytic mode that includes full attention-core FLOPs (and other torch-unseen work).

Given the above, the user’s updated preferences are:

1. In pipeline verification scripts, keep the vendor model using **eager LLaMA attention** (no FlashAttention2) to avoid introducing new undercounting artifacts on the vendor side.
2. For the main verification sanity checks, configure the analytic model to match the vendor’s eager attention and set `ignore_torch_unsupported_flop_count=False` (count everything analytic); then compare full analytic FLOPs to the eager vendor FLOPs.
3. For the analytic model only, also compute a **flash-attention analytic variant**, and report those FLOPs as an additional curve in sweeps/verification results. This variant is not expected to match vendor numbers; it is there to show how a hypothetical FlashAttention2-based analytic path behaves.

The current scripts do not clearly separate these three concerns. We need a structured, explicit scheme instead of scattered flags and implicit assumptions.


How to Refactor
---------------

### 1. Make vendor attention mode explicit (and keep it eager)

- Introduce a helper in `run_verify_end2end.py` (or a small shared module in `scripts/verify`) to construct the vendor OCR model with **eager** attention only:

  ```python
  def build_vendor_ocr_model_eager(
      vendor: _VendorModules,
      model_root: Path,
      device: torch.device,
  ) -> nn.Module:
      config = vendor.DeepseekOCRConfig.from_pretrained(str(model_root))
      # Always use eager attention for verification to keep vendor FLOPs
      # fully visible to FlopCounterMode.
      config._attn_implementation = "eager"
      model = vendor.DeepseekOCRForCausalLM(config)
      model.to(device)
      model.eval()
      return model
  ```

- Replace all direct uses of `_build_reference_ocr_model(...)` in:
  - `run_verify_end2end.py`
  - `run_verify_end2end_prefill_decode.py`
  - `run_verify_prefill_decode.py`
  - `run_verify_core.py`
  with `build_vendor_ocr_model_eager(...)`.
- The goal is to codify that, for pipeline FLOP verification, the vendor path is **always** the eager implementation. Any FlashAttention2 experiments remain strictly analytic.

### 2. Define analytic FLOP modes and use them explicitly

- Introduce a small enum or type for analytic modes, e.g. in a shared helper (under `scripts/verify` or `layers/core`):

  ```python
  from enum import Enum

  class AnalyticFlopMode(Enum):
      FULL_EAGER = "full_eager"        # eager attention, ignore_torch_unsupported_flop_count=False
      FLASH_FULL = "flash_full"        # flash attention, ignore_torch_unsupported_flop_count=False
      # (optional) FLASH_TORCH_VISIBLE = "flash_torch" if we ever want to approximate torch-visible under flash.
  ```

- Refactor the stage-level analytic computation helpers (for example, in `run_verify_end2end_prefill_decode.py` and `sweep-e2e-*.py`):

  ```python
  def compute_analytic_prefill_decode_tflops(
      cfg: DictConfig,
      *,
      context_len: int,
      batch_size: int,
      num_decode_steps: int,
      mode: AnalyticFlopMode,
  ) -> tuple[float, float]:
      cfg_local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))

      if mode is AnalyticFlopMode.FLASH_FULL:
          cfg_local.model.decoder_attn_impl = "flash_attention_2"
      else:
          cfg_local.model.decoder_attn_impl = "eager"

      analytic_model = build_analytic_model(cfg_local)
      # IMPORTANT: For pipeline verification we always use FULL mode here:
      # ignore_torch_unsupported_flop_count=False. The goal is for analytic
      # FLOPs to reflect the full modeled cost (including ops that PyTorch's
      # flop counter might miss), while the vendor path remains eager LLaMA
      # attention measured by FlopCounterMode.
      analytic_model._set_ignore_torch_unsupported_flop_count(False)
      ...
  ```

- For verification comparisons, always compute at least:
  - `analytic_full_eager` (baseline: matches vendor eager implementation),
  - `analytic_full_flash` (hypothetical FlashAttention2 analytic path).

### 3. Verification scripts: print both eager and flash analytic FLOPs

- In `run_verify_prefill_decode.py` and `run_verify_end2end_prefill_decode.py`, restructure the summary output as:

  ```text
  Prefill FLOPs (TFLOPs):
    vendor (eager, torch)        : <prefill_vendor_tflops>
    analytic eager (full)        : <prefill_analytic_eager_tflops>
    analytic flash (full)        : <prefill_analytic_flash_tflops>

  Decode FLOPs (TFLOPs):
    vendor (eager, torch)        : <decode_vendor_tflops>
    analytic eager (full)        : <decode_analytic_eager_tflops>
    analytic flash (full)        : <decode_analytic_flash_tflops>
  ```

- Emphasize in docstrings and help text:
  - Vendor FLOPs use eager attention and represent what `FlopCounterMode` sees for that path.
  - `analytic eager (full)` is the primary sanity check against vendor FLOPs.
  - `analytic flash (full)` is a “forward-looking” analytic scenario that does **not** match vendor FLOPs but illustrates how a FlashAttention2-based analytic path would look.

### 4. Sweep scripts: add flash analytic curves

- In `sweep-e2e-crops.py`, `sweep-e2e-vision-prefill.py`, and `sweep-e2e-decode.py`, extend the stage dictionaries to capture both eager and flash analytic variants:

  ```python
  "decode": {
      "analytic_eager_tflops": float(decode_full_eager),
      "analytic_flash_tflops": float(decode_full_flash),
      "vendor_tflops": float(decode_vendor_tflops) if decode_vendor_tflops is not None else None,
  }
  ```

- Plots:
  - Update labels to distinguish:
    - `analytic (eager, full)`
    - `analytic (flash, full)`
    - `vendor (eager, torch)`
  - The eager analytic curve is the primary match to the vendor line; the flash analytic curve is an additional series on the same axes, visually showing the potential behavior under FlashAttention2.

### 5. Example before/after snippets

**Before (implicit modes, only one analytic variant):**

```python
# Vendor model: eager, implicit
config = vendor.DeepseekOCRConfig.from_pretrained(str(model_root))
ref_model = vendor.DeepseekOCRForCausalLM(config).to(device).eval()

# Analytic: single mode, often with ignore_torch_unsupported_flop_count=True
analytic_model._set_ignore_torch_unsupported_flop_count(True)
prefill_tflops, decode_tflops = _compute_analytic_end_to_end_prefill_decode_tflops(
    analytic_model,
    context_len=context_len,
    batch_size=batch_size,
    num_decode_steps=decode_steps,
)
print("decode: vendor", decode_ref_tflops, "analytic", decode_tflops)
```

**After (explicit eager vendor + eager/flash analytic modes):**

```python
ref_model = build_vendor_ocr_model_eager(
    vendor=vendor,
    model_root=model_root,
    device=torch_device,
)

prefill_eager, decode_eager = compute_analytic_prefill_decode_tflops(
    cfg_base,
    context_len=context_len,
    batch_size=batch_size,
    num_decode_steps=decode_steps,
    mode=AnalyticFlopMode.FULL_EAGER,
)
prefill_flash, decode_flash = compute_analytic_prefill_decode_tflops(
    cfg_base,
    context_len=context_len,
    batch_size=batch_size,
    num_decode_steps=decode_steps,
    mode=AnalyticFlopMode.FLASH_FULL,
)

print("Decode FLOPs (TFLOPs):")
print(f"  vendor (eager, torch)       : {decode_ref_tflops:.3f}")
print(f"  analytic eager (full)       : {decode_eager:.3f}")
print(f"  analytic flash (full)       : {decode_flash:.3f}  [no vendor match]")
```


Impact Analysis
---------------

**Functional impact**

- Vendor side:
  - Behavior remains eager attention, matching the current deployment path and current flop-counter behavior.
  - FLOP numbers reported for vendor runs stay comparable to existing runs (modulo minor refactors).
- Analytic side:
  - We will now compute two analytic decode/prefill curves:
    - Eager analytic (full), used for sanity checking against vendor.
    - Flash analytic (full), a hypothetical variant that may have different scaling.
  - Sweep outputs (JSON, plots) will gain new fields and legend entries for flash analytic curves.

**Risks**

- Downstream tooling that expects a single analytic curve may need to be updated to select `analytic_eager_tflops` explicitly.
- If we rename existing keys (`analytic_tflops`), we might break older scripts.

**Mitigation**

- Keep existing keys where possible:
  - For sweeps, we can alias:
    - `analytic_tflops` → `analytic_eager_tflops` for backward compatibility.
  - Add `analytic_flash_tflops` as a new field.
- Document the change clearly in:
  - `scripts/README.md`
  - `docs/about-torch-flop-counter-and-analytic-modes.md`
  - `docs/caveats/caveats-torch-flop-counter.md`
- In verification scripts, continue to emphasize the eager analytic vs vendor eager comparison as the main sanity check.


Expected Outcome
----------------

Once this refactor is in place:

- The vendor model used in all e2e verification scripts will clearly and consistently use eager LLaMA attention.
- Verification outputs will always show:
  - Vendor FLOPs (eager, torch).
  - Analytic eager FLOPs (full, matching vendor attention semantics).
  - Analytic flash FLOPs (full, hypothetical, not calibrated to vendor).
- Sweeps will expose:
  - `analytic_eager_tflops` and `analytic_flash_tflops` curves for each stage.
  - `vendor_tflops` curves where available.

This gives us:

- A clean sanity-check path (eager analytic vs eager vendor under torch’s flop counter).
- A way to visualize and reason about alternative attention implementations (FlashAttention2) in the analytic model, even when they do not match vendor FLOP numbers.


References
----------

- Vendor model and attention implementation:
  - `models/deepseek-ocr/modeling_deepseekv2.py`
  - `models/deepseek-ocr/configuration_deepseek_v2.py`
- Verification scripts:
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_end2end_prefill_decode.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_prefill_decode.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/verify/run_verify_core.py`
- Sweep scripts:
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-crops.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-vision-prefill.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-decode.py`
- FLOP counter caveats and analytic modes:
  - `extern/modelmeter/models/deepseek_ocr/docs/caveats/caveats-torch-flop-counter.md`
  - `extern/modelmeter/models/deepseek_ocr/docs/about-torch-flop-counter-and-analytic-modes.md`
- Debug / verification helpers:
  - `extern/modelmeter/models/deepseek_ocr/scripts/misc/verify_torch_flop_count_behaviour.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/misc/verify_flash_attention_flops.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/misc/verify_llama_attention.py`
- Third-party library (for reference):
  - Hugging Face Transformers (LLaMA + FlashAttention2 integration):
    - Context7 library id: `/huggingface/transformers`
