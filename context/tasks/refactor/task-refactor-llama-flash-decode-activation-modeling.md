## What to Refactor
Refactor the decode activation and I/O modeling for the FlashAttention2 analytic path used in DeepSeek OCR so that per token decode metrics match the intended semantics and align with the non flash (eager) attention path.
Concretely, this focuses on:
- `extern/modelmeter/models/deepseek_ocr/layers/llama/llama_flash_attention2.py`:
  - `LlamaFlashAttention2.set_decode_shape`
  - `LlamaFlashAttention2.forward_cal_io`
  - `LlamaFlashAttention2.forward_memory_activation`
- Downstream consumers of `StageCost` for decode:
  - `extern/modelmeter/models/deepseek_ocr/layers/decoder/deepseek_v2_decoder_layer.py` (decode mode, Flash vs non Flash attention)
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py` (decode stage aggregation)
  - Sweep and reporting scripts that surface `decode_stagecost`, especially:
    - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-decode.py`
    - `extern/modelmeter/models/deepseek_ocr/scripts/reporting/kvcache_by_input_shape.py`

## Why Refactor
Today, `decode_stagecost.decode_flash.activations_gb` is much larger than `decode_stagecost.decode_eager.activations_gb` for the same workload in end to end decode sweeps.
This is counterintuitive because both modes are modeling the same logical decoder stack and differ only in how attention is implemented (FlashAttention2 vs standard SDPA), which should not change activation memory by orders of magnitude for per token decode.
The root cause is that `LlamaFlashAttention2.forward_memory_activation` still uses the legacy single `seq_len` (`self.m_seq_len`) as its sequence length, and `set_decode_shape` sets `self.m_seq_len` to `context_len` even for decode.
As a result, the Flash path effectively treats the full context length as if it were the query length when estimating activation memory, whereas the eager path already uses `S_q = 1` for decode and only uses `S_kv` for KV cache.
This discrepancy breaks the intended semantics described in `HOLISTIC_ANALYSIS.md`, where:
- Prefill activations represent a context wide forward pass.
- Decode activations represent per token decoder activations on top of an existing KV cache and are expected to be much smaller than prefill.
Keeping the current behavior:
- Misleads capacity and MFU analysis that relies on `decode_stagecost.decode_flash.activations_gb` to reason about per token decode memory.
- Introduces a large gap between analytic flash vs eager paths that is purely an artifact of the modeling implementation rather than the underlying hardware behavior.
Refactoring the Flash decode modeling to be query/context aware and aligned with the eager path will make activation and I/O metrics more trustworthy and easier to interpret in reports and plots.

## How to Refactor

### 1) Clarify desired decode semantics
- For decode, both eager and flash attention paths should model:
  - FLOPs that depend on both query length (`S_q`) and context length (`S_kv`).
  - Activation I/O and activation memory that are dominated by the new query token(s), with only compact context dependent state when appropriate.
- For `StageCost` in decode:
  - `flops_tflops` should represent total FLOPs per token, including attention core and projections.
  - `io_tb` should reflect I/O for processing the new token given the current KV state.
  - `activations_gb` should approximate per token decoder activations for the current token, not the full context.
  - `kv_gb` should account for the KV cache footprint at the current context length (prefill plus previous decode tokens).
Ensure this contract is explicitly documented in `HOLISTIC_ANALYSIS.md` and matches how `DeepseekOCRModel._compute_analytic_decode_stage_cost` aggregates per step costs.

### 2) Make FlashAttention2 decode semantics query/context aware
In `extern/modelmeter/models/deepseek_ocr/layers/llama/llama_flash_attention2.py`:
- Confirm that `set_decode_shape` is correctly setting:
  - `self.m_seq_len_q = 1`
  - `self.m_seq_len_kv = context_len`
  - `self.m_seq_len = context_len` (legacy field)
- Refactor `forward_memory_activation` to use query and context lengths explicitly:
  - For prefill, keep the current interpretation where `S = S_q = S_kv`.
  - For decode, ensure that:
    - Input, QKV, and output activations scale with `S_q` (usually 1).
    - Any additional compact attention buffer uses `S_q` rather than `S_kv` to avoid reintroducing full `S_kv × S_kv` style scaling.
- Refactor `forward_cal_io` if needed:
  - Today it already uses `self.m_seq_len_q` as `S` for both prefill and decode.
  - Verify that this is consistent with the discussion in `LAYER_IO_ESTIMATION_GUIDE.md` and adjust if we should additionally reflect some context length dependent reads (for example, through the compact FlashAttention2 buffers).
Target shape aware implementation sketch (Flash decode path):

```python
# Before (simplified)
def forward_memory_activation(self) -> float:
    b = float(self.m_batch_size)
    s = float(self.m_seq_len)  # decode: context_len
    d_model = float(self.m_hidden_size)
    bytes_per_val = 2.0
    input_bytes = b * s * d_model * bytes_per_val
    qkv_bytes = b * s * (3.0 * d_model) * bytes_per_val
    attn_bytes = b * s * d_model * bytes_per_val
    output_bytes = b * s * d_model * bytes_per_val
    total_bytes = input_bytes + qkv_bytes + attn_bytes + output_bytes
    return total_bytes / (1024.0 ** 3)
```

```python
# After (high level idea)
def forward_memory_activation(self) -> float:
    b = float(self.m_batch_size)
    d_model = float(self.m_hidden_size)
    bytes_per_val = 2.0
    if self.m_stage == "prefill":
        s = float(self.m_seq_len_q)  # full context
    else:
        s = float(self.m_seq_len_q)  # decode: query length (≈ 1)
    input_bytes = b * s * d_model * bytes_per_val
    qkv_bytes = b * s * (3.0 * d_model) * bytes_per_val
    attn_bytes = b * s * d_model * bytes_per_val
    output_bytes = b * s * d_model * bytes_per_val
    total_bytes = input_bytes + qkv_bytes + attn_bytes + output_bytes
    return total_bytes / (1024.0 ** 3)
```

The exact formulas can reuse the existing ones but must explicitly pick `s` based on the stage and shape (`seq_len_q` vs `seq_len_kv`).

### 3) Verify eager vs flash decoder layer aggregation
`DeepseekV2DecoderLayer.forward_memory_activation` currently aggregates RMSNorm, attention, and MLP contributions:

```python
norm1_mem = self.m_input_layernorm.forward_memory_activation()
attn_mem = self.m_self_attn.forward_memory_activation()
norm2_mem = self.m_post_attention_layernorm.forward_memory_activation()
mlp_mem = self.m_mlp.forward_memory_activation()
return (norm1_mem or 0.0) + (attn_mem or 0.0) + (norm2_mem or 0.0) + (mlp_mem or 0.0)
```

With the FlashAttention2 fix in place:
- For prefill, `attn_mem` will still scale with full context length and match (up to modeling differences) the eager path.
- For decode, `attn_mem` will scale with query length, and `norm*` and `mlp` already operate on `seq_len = 1`, so the overall decoder activation footprint per token should be on the same order for eager and flash.
As part of the refactor:
- Compare `forward_memory_activation()` for a representative decoder layer in:
  - Prefill mode vs decode mode for both eager and flash.
  - Flash vs eager in decode mode across a few context lengths (for example, 1k, 4k, 8k) to ensure the ratio stays within a reasonable band (for example, within a small constant factor).

### 4) Update holistic documentation and design notes
Update the relevant docs to reflect the new behavior:
- `extern/modelmeter/models/deepseek_ocr/HOLISTIC_ANALYSIS.md`:
  - Clarify that FlashAttention2 decode uses per token activation modeling similar to eager attention.
  - Update or remove the bullet that says `LlamaFlashAttention2.forward_cal_io()` and `forward_memory_activation()` are not query/context aware.
- `extern/modelmeter/models/deepseek_ocr/LAYER_IO_ESTIMATION_GUIDE.md`:
  - Document how `LlamaFlashAttention2` handles `S_q` and `S_kv` in FLOPs, I/O, activations, and KV cache for prefill vs decode.
- Any open TODOs in `context/tasks/refactor/task-refactor-dsocr-analytic-prefill-decode-apis.md` that refer to the old single `seq_len` behavior should be updated to mark this aspect as completed.

### 5) Adjust sweep and reporting scripts if needed
Most sweep code consumes `StageCost` as an opaque structure and will automatically benefit from the corrected Flash decode modeling, but we should review:
- `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-decode.py`:
  - Sanity check that `decode_stagecost.decode_flash.activations_gb` now behaves as expected (per token, smaller than prefill, roughly aligned with eager).
  - Optionally add a small assertion or logging check guarded by a flag to detect obviously wrong ratios during development (for example, flash decode activations exceeding prefill by orders of magnitude).
- `extern/modelmeter/models/deepseek_ocr/scripts/reporting/kvcache_by_input_shape.py`:
  - Confirm that any derived metrics or plots that use `decode_stagecost["decode_flash"].activations_gb` still make sense after the change.
- Any Stage 1 or Stage 2 analyzers in `src/llm_perf_opt/runners/dsocr_analyzer.py` that explicitly mention flash vs eager decode memory should be updated to describe the new behavior if they rely on these fields.

### 6) Add and update tests
Augment `tests/unit/deepseek_ocr/test_analytic_layers_scaling.py` and related tests to cover:
- For `LlamaFlashAttention2`:
  - Prefill: activation memory increases with sequence length as expected.
  - Decode: activation memory is roughly constant with respect to context length and scales with `S_q` (for example, compare `S_q = 1` vs `S_q = 4` if supported).
  - Decode: flash vs eager activation memory for the same context length and token count differ by at most a modest factor (for example, < 2× for representative shapes).
- For `DeepseekV2DecoderLayer`:
  - Prefill vs decode activation memory behave as described in `HOLISTIC_ANALYSIS.md` (decode << prefill for large contexts).
Add integration level checks in tests that exercise the sweep scripts:
- Use a small candidate shape DB and verify that:
  - For at least one point, `decode_stagecost.decode_flash.activations_gb` is not orders of magnitude larger than `decode_stagecost.decode_eager.activations_gb`.
  - Decode activations are much smaller than prefill activations for the same shape.
These tests should avoid hard coding exact numeric values and instead check monotonicity and reasonable ratios to remain robust to future analytic tweaks.

### 7) Migration and compatibility considerations
- This change will alter numeric values in:
  - Existing sweep JSON artifacts under `reports/` and `tmp/profile-output/`.
  - Derived SQLite databases and plots that include Flash decode activation metrics.
- To keep downstream consumers safe:
  - Call out the change in any release notes or analyzer documentation that mention decode memory modeling, with a brief explanation that Flash decode activations moved from full context based to per token modeling.
  - Consider bumping a small schema or version field in generated sweep JSON (for example, adding `"analytic_model_version": "2024-XX-flash-decode-fix"`) so that external analysis scripts can detect which convention applies if necessary.

## Impact Analysis

### Functional impact
- Flash decode activation estimates (`decode_stagecost.decode_flash.activations_gb`) will drop significantly for typical large contexts because they will no longer scale with full context length.
- Prefill activation estimates and KV cache footprints are expected to remain unchanged because:
  - Prefill still uses full context length for both eager and flash paths.
  - KV modeling is already context length aware and uses `seq_len_kv` in both attention implementations.
- FLOP counts (`analytic_flash_tflops`, `decode_flash.flops_tflops`) are not expected to change because they already use `S_q` and `S_kv` appropriately.

### Downstream tools and reports
- Plots generated by `sweep-e2e-decode.py` and `kvcache_by_input_shape.py` that visualize activation memory vs image tokens will show:
  - Flash decode activation curves much closer to the eager decode curves.
  - Decode activation curves well below prefill curves, matching the narrative in `HOLISTIC_ANALYSIS.md`.
- Any dashboards or notebooks that previously assumed flash decode activations were unrealistically large may need to be reinterpreted but will not break structurally.

### Risks
- Existing analyses that were calibrated or fit against the old Flash decode activation numbers may see noticeable shifts in derived metrics (for example, memory headroom estimates, MFU approximations that indirectly use activation memory as a proxy).
- If any external code relies on the absolute value of `activations_gb` for Flash decode, it could diverge from expectations after this change.
- There is a small risk of introducing inconsistencies if `forward_memory_activation` and `forward_cal_io` use different sequence lengths for decode, so they must be updated together and documented.

### Mitigations
- Keep eager attention behavior unchanged and use it as a reference during testing; for representative shapes, verify that:
  - Flash decode activations are within a reasonable multiple of eager decode activations.
  - Ratios are stable across context lengths.
- Add targeted unit tests and integration checks as described above so regressions are caught early.
- Clearly document the behavior change and, if needed, introduce a lightweight version tag in sweep outputs so downstream consumers can branch on the analytic model version if they must.

## Expected Outcome
- `decode_stagecost.decode_flash.activations_gb` will represent per token decoder activations for FlashAttention2 decode, aligned with the eager attention path and the design described in `HOLISTIC_ANALYSIS.md`.
- Decode activation curves for flash and eager will become comparable and will sit well below prefill activations for large contexts, supporting intuitive reasoning about memory usage over prefill vs decode.
- I/O and activation modeling for FlashAttention2 will be clearly query/context aware and documented, reducing confusion when interpreting end to end DeepSeek OCR analytic reports.
- Downstream analysis tools in `llm_perf_opt` (for example, `dsocr_analyzer` and reporting scripts) will be able to trust `StageCost.activations_gb` for Flash decode when making capacity and MFU related decisions.

## References
- Code:
  - `extern/modelmeter/models/deepseek_ocr/layers/llama/llama_flash_attention2.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/llama/llama_attention.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/decoder/deepseek_v2_decoder_layer.py`
  - `extern/modelmeter/models/deepseek_ocr/layers/core/deepseek_ocr_model.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/sweep/sweep-e2e-decode.py`
  - `extern/modelmeter/models/deepseek_ocr/scripts/reporting/kvcache_by_input_shape.py`
- Docs and design notes:
  - `extern/modelmeter/models/deepseek_ocr/HOLISTIC_ANALYSIS.md`
  - `extern/modelmeter/models/deepseek_ocr/LAYER_IO_ESTIMATION_GUIDE.md`
  - `extern/modelmeter/models/deepseek_ocr/LAYER_IMPL_GUIDE.md`
  - `context/tasks/refactor/task-refactor-dsocr-analytic-prefill-decode-apis.md`
- Tests:
  - `tests/unit/deepseek_ocr/test_analytic_layers_scaling.py`
  - `tests/unit/deepseek_ocr/test_vision_flash_attention_variants.py`
- Third party libraries (Context7 IDs):
  - `/pytorch/pytorch`
  - `/huggingface/transformers`

