# Review: DeepSeek-OCR Prefill/Decode Separation Guide

**Document Reviewed**: `howto-separate-prefill-decode-deepseek-ocr.md`
**Review Date**: 2025-01-05
**Reviewer**: Claude Code
**Overall Assessment**: ✅ **Largely Correct** (8.5/10)

---

## Executive Summary

The document correctly describes the architectural pattern for separating prefill and decode phases in DeepSeek-OCR profiling. The core insight—that image processing is gated by `seq_len != 1`—is accurate and represents the critical mechanism enabling this separation. Code examples are practical and should work with minor adjustments.

**Key Strengths**:
- Sound architectural understanding of vision-language model phases
- Practical, runnable code examples
- Good coverage of edge cases (dtype, attention masks, BOS/EOS)

**Key Weaknesses**:
- Line numbers will drift without version pinning
- Missing validation/diagnostic code
- Some ambiguities in edge case handling

---

## Detailed Findings

### ✅ Correct Architectural Understanding

**Core Mechanism** (Line 403 gate):
```python
if images is not None and seq_len != 1:
    # Visual processing happens here
```

- **Prefill**: `seq_len > 1` → visual encoder runs, features injected, KV cache built
- **Decode**: `seq_len == 1` → visual encoder skipped, cache reused

This is the correct pattern for vision-language models with KV caching.

**Code Flow References** (as documented):
- Image embedding gate: Line 403
- Visual feature computation: Lines 422, 426-464, 505
- Base transformer delegation: Lines 510-515
- Input preparation: Lines 620-690
- Example generate call: Lines 916-929
- Cache handling: Lines 1540-1574, 1607-1637

---

## Issues & Ambiguities

### ⚠️ **Issue 1: Line Numbers Will Drift**

**Problem**: Hardcoded line numbers (403, 422, etc.) without version pinning will become stale as code evolves.

**Impact**: Medium (maintainability)

**Recommendation**:
Add a header to the document:
```markdown
**Code Version**: Commit `<hash>` or Tag `v1.0.0`
**Files Referenced**:
- `models/deepseek-ocr/modeling_deepseekocr.py`
- `models/deepseek-ocr/modeling_deepseekv2.py`
**Last Verified**: 2025-01-05
```

**Verification Command**:
```bash
git log -1 --oneline models/deepseek-ocr/modeling_deepseekocr.py
```

---

### 🤔 **Issue 2: First Token Generation Logic**

**Code in Question** (Lines 65-66):
```python
next_input_ids = input_ids[:, -1:].contiguous()  # [B, 1]
```

**Analysis**:
This re-decodes the last input token to generate the first new token. However, `prefill_logits[:, -1, :]` already contains the distribution for the first generated token.

**Two Valid Approaches**:

**Option A (Current - Re-decode first token)**:
```python
# Good for profiling pure decode behavior
next_input_ids = input_ids[:, -1:].contiguous()
```
✅ Measures decode-only latency accurately
❌ Redundant computation (already in prefill_logits)

**Option B (Use prefill logits)**:
```python
# More efficient for production
first_token = torch.argmax(prefill_logits[:, -1, :], dim=-1).unsqueeze(1)
next_input_ids = first_token
```
✅ No redundant computation
❌ Doesn't measure pure decode for first token

**Verdict**: For **profiling purposes**, Option A (current) is acceptable since you want to measure decode-only latency. For **production inference**, Option B is more efficient.

---

### 📝 **Issue 3: `prepare_inputs_for_generation` Contract**

**Assumption** (Lines 76-82):
The decode loop assumes `prepare_inputs_for_generation` correctly handles:
- Omitted image args (not passed → should return None or omit from dict)
- Position ID computation with cache
- Input slicing based on cache length

**Verification Needed**:
```bash
# Check OCR variant implementation (lines 620-690)
grep -A 50 "def prepare_inputs_for_generation" \
  models/deepseek-ocr/modeling_deepseekocr.py
```

**Expected Behavior**:
- If `images` kwarg is not passed, it should not appear in returned dict
- `position_ids` should account for cache length
- `input_ids` should be correctly sliced to `[B, 1]`

**Recommended Validation**:
```python
prepared = model.prepare_inputs_for_generation(...)
assert 'images' not in prepared, "Image args leaked into decode step"
assert prepared['input_ids'].shape[1] == 1, "Input should be single token"
```

---

### ⚠️ **Issue 4: EOS Handling is Simplified**

**Code** (Line 99):
```python
if eos_id is not None and (next_token == eos_id).all().item():
    break
```

**Problem**: Stops when **all** sequences hit EOS. Sequences that finish early continue generating padding.

**Impact**: Low (for profiling with fixed `max_new_tokens`, acceptable)

**Production-Ready Alternative**:
```python
finished = torch.zeros(B, dtype=torch.bool, device=device)
# ... in loop:
finished |= (next_token == eos_id)
if finished.all():
    break

# Mask out finished sequences
next_token = torch.where(finished.unsqueeze(1), pad_token_id, next_token)
```

**Verdict**: ✅ Current approach is fine for profiling. Document should note this is simplified.

---

### ⚠️ **Issue 5: Attention Mask Initialization**

**Code** (Line 68):
```python
attn_mask = attention_mask.clone()  # starts as prefill mask [B, T]
```

**Assumptions**:
- `attention_mask` from prefill has shape `[B, T]` (not 4D causal mask)
- Mask is already on correct device
- Cloning is sufficient (no in-place ops on original)

**Verification Needed**:
```python
print(f"Attention mask shape: {attention_mask.shape}")
print(f"Attention mask device: {attention_mask.device}")
assert len(attention_mask.shape) == 2, "Expected 2D mask"
assert attention_mask.device == input_ids.device, "Device mismatch"
```

---

## Potential Bugs to Watch

### 🐛 **Bug Risk 1: KV Cache Memory Leak**

**Scenario**: If `past_key_values` grows unboundedly due to incorrect slicing in `prepare_inputs_for_generation`, you'll OOM.

**Diagnostic Code**:
```python
if _ == 0:
    # First decode step - record initial cache size
    initial_cache_len = past_kv[0][0].shape[2]  # seq_len dimension
    print(f"Initial cache length: {initial_cache_len}")

# After each step
current_cache_len = past_kv[0][0].shape[2]
expected_len = initial_cache_len + (_ + 1)
assert current_cache_len == expected_len, \
    f"Cache length mismatch: {current_cache_len} != {expected_len}"
```

**Expected Behavior**:
- Initial cache length: `T` (prefill sequence length)
- After step `n`: `T + n`

---

### 🐛 **Bug Risk 2: Image Arg Leakage**

**Scenario**: Image args accidentally passed during decode → visual encoder runs unnecessarily.

**Diagnostic Code**:
```python
prepared = model.prepare_inputs_for_generation(
    next_input_ids,
    past_key_values=past_kv,
    attention_mask=attn_mask,
    use_cache=True,
)

# Defensive checks
assert 'images' not in prepared, "Image args leaked into decode"
assert 'images_seq_mask' not in prepared, "Image mask leaked into decode"
assert 'images_spatial_crop' not in prepared, "Image spatial crop leaked into decode"
```

**Profiler Verification**:
```python
# During decode, visual encoder should show 0 time
# Check in PyTorch profiler output or Nsight Systems timeline
```

---

### 🐛 **Bug Risk 3: Autocast Scope Issues**

**Code** (Lines 35, 84):
```python
with torch.autocast("cuda", dtype=torch.bfloat16):
```

**Potential Issue**: If some model components expect fp32 (e.g., LayerNorm), autocast may cause numerical instability.

**Verification**:
```python
# Check if model has autocast-sensitive ops
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
        print(f"Found norm layer: {name}")
```

**Recommendation**: Use model's native dtype if available:
```python
# If model was loaded with torch_dtype=torch.bfloat16
with torch.no_grad():  # autocast may not be needed
    outputs = model(...)
```

---

## Missing Content

### 📋 **Missing: Verification Section**

**Add to Document**:
```markdown
## Verification Checklist

After implementing prefill/decode separation:

1. **Functional Correctness**
   - [ ] Decode-generated tokens match one-shot `generate()` output
   - [ ] No image args leak into decode steps
   - [ ] KV cache size grows by exactly 1 token per step

2. **Performance Validation**
   - [ ] Prefill includes vision encoder time (5-10× slower than decode)
   - [ ] Decode per-token time is consistent across steps
   - [ ] Visual encoder shows 0% time during decode in profiler

3. **Memory Safety**
   - [ ] No OOM during long sequences
   - [ ] Peak memory during decode ≤ peak during prefill
   - [ ] Cache size grows linearly with generated tokens

**Verification Script**:
```python
# Run both methods and compare outputs
one_shot_ids = model.generate(...)
prefill_decode_ids = run_prefill_decode_separate(...)
assert torch.equal(one_shot_ids, prefill_decode_ids), "Output mismatch!"
```
```

---

### 📋 **Missing: Performance Expectations**

**Add to Document**:
```markdown
## Performance Expectations

**Typical Timings** (for reference, hardware-dependent):
- **Prefill**: 50-200ms (includes vision encoder + LLM forward)
- **Decode**: 5-20ms per token (LLM only, no vision)
- **Ratio**: Prefill should be 5-15× slower than single decode step

**Profiling Breakdown**:
- Visual encoder: ~30-50% of prefill time
- Text LLM forward (prefill): ~40-60% of prefill time
- KV cache write: ~5-10% of prefill time
- Text LLM forward (decode): ~90-95% of decode time
- Cache read/update: ~5-10% of decode time

**Red Flags**:
- ⚠️ Visual encoder shows non-zero time during decode
- ⚠️ Decode time increases with sequence length (should be constant)
- ⚠️ Prefill time ≈ decode time (visual encoder not running)
```

---

### 📋 **Missing: Troubleshooting Section**

**Add to Document**:
```markdown
## Troubleshooting

### Visual Encoder Runs During Decode
**Symptom**: Decode time unexpectedly high, profiler shows vision module active

**Causes**:
1. `images` arg accidentally passed during decode
2. `seq_len != 1` check failing (input shape wrong)
3. Model in training mode (some conditional logic may differ)

**Fix**:
```python
model.eval()  # Ensure eval mode
assert 'images' not in prepared, "Check prepare_inputs_for_generation"
assert next_input_ids.shape[1] == 1, f"Expected [B,1], got {next_input_ids.shape}"
```

### OOM During Decode
**Symptom**: Memory usage grows unboundedly

**Causes**:
1. KV cache not reused (new cache created each step)
2. Attention mask growing incorrectly (wrong dimension)
3. Gradient tracking enabled

**Fix**:
```python
with torch.no_grad():  # Disable gradient tracking
    ...
# Verify cache reuse:
print(f"Cache ID: {id(past_kv)} (should stay constant)")
```

### Output Mismatch vs One-Shot Generate
**Symptom**: Prefill/decode output differs from `generate()`

**Causes**:
1. Sampling vs greedy (temperature, top_k/top_p)
2. Different tokenization (BOS/EOS handling)
3. Attention mask construction error

**Fix**:
```python
# Use greedy decoding for exact comparison
next_token = torch.argmax(step_logits, dim=-1)  # NOT sampling
# Ensure same tokenization
assert torch.equal(input_ids, one_shot_input_ids), "Input mismatch"
```
```

---

## Recommended Additions

### 1. **Version Pinning Header**
```markdown
**Code Version**: Commit `abc123` (2025-01-05)
**Files**:
- `models/deepseek-ocr/modeling_deepseekocr.py` (lines 403, 422, ...)
- `models/deepseek-ocr/modeling_deepseekv2.py` (lines 1540, 1607, ...)

**Last Verified**: 2025-01-05
```

### 2. **Line 403 Condition Clarification**
Replace:
> "The image branch is gated off because `seq_len == 1`"

With:
```markdown
**Line 403 Gate Condition**:
```python
if images is not None and seq_len != 1:
    # Visual processing happens here
```

**Behavior**:
- **Prefill**: `seq_len > 1` (e.g., 256 tokens) → Gate OPEN → Visual features computed
- **Decode**: `seq_len == 1` (single token) → Gate CLOSED → Visual branch skipped
```

### 3. **Diagnostic Code Snippets**
Add inline validation examples to prefill/decode code blocks (see Bug Risks section above).

---

## Integration with Existing Codebase

### Check Existing Implementation

**File to Review**: `src/llm_perf_opt/runners/dsocr_session.py`

**Questions**:
1. Does `DeepSeekOCRSession` already implement prefill/decode separation?
2. Does it follow the pattern described in the document?
3. Are there additional edge cases handled in the session wrapper?

**Verification Command**:
```bash
grep -n "past_key_values\|prepare_inputs_for_generation" \
  src/llm_perf_opt/runners/dsocr_session.py
```

**Expected Alignment**:
- Session wrapper should encapsulate prefill/decode logic
- Should expose methods like `run_prefill()`, `run_decode_step()`
- Should include validation/assertions per this review

---

## Action Items

### High Priority
- [ ] Add version/commit pinning to document header
- [ ] Verify line 403 gate condition in current code
- [ ] Add KV cache growth validation to decode loop
- [ ] Add image arg leakage checks

### Medium Priority
- [ ] Add verification checklist section
- [ ] Add performance expectations section
- [ ] Add troubleshooting section
- [ ] Clarify first token generation options (current vs. alternative)

### Low Priority
- [ ] Add EOS handling note (simplified for profiling)
- [ ] Document autocast scope considerations
- [ ] Cross-reference with `dsocr_session.py` implementation

---

## Verification Commands

### 1. Verify Line 403 Gate
```bash
grep -n "images is not None and.*seq_len" \
  models/deepseek-ocr/modeling_deepseekocr.py
```

**Expected Output**:
```
403:        if images is not None and seq_len != 1:
```

### 2. Check `prepare_inputs_for_generation`
```bash
grep -A 70 "def prepare_inputs_for_generation" \
  models/deepseek-ocr/modeling_deepseekocr.py | head -75
```

**Verify**:
- Handles missing `images` arg gracefully
- Computes `position_ids` using cache length
- Returns dict without image args when omitted

### 3. Verify Base Transformer Cache Handling
```bash
grep -n "past_key_values.*update\|cache.*update" \
  models/deepseek-ocr/modeling_deepseekv2.py | head -20
```

**Verify**:
- Cache is updated in-place or returned correctly
- Cache shape follows `(batch, num_heads, seq_len, head_dim)` convention

---

## Final Recommendations

### For Document Maintainer
1. **Add version header** with commit hash and verification date
2. **Add verification section** with checklist and diagnostic code
3. **Add troubleshooting section** covering common issues
4. **Note simplifications** (EOS handling, first token logic) explicitly

### For Document User
1. **Verify line numbers** before relying on references
2. **Add validation code** (cache growth, image leakage) to your implementation
3. **Cross-check with profiler** that visual encoder is inactive during decode
4. **Compare output** with one-shot `generate()` to ensure correctness

### For Codebase Integration
1. **Review `dsocr_session.py`** for alignment with this pattern
2. **Implement validation assertions** in session wrapper
3. **Add unit tests** comparing prefill/decode with one-shot generation
4. **Document in CLAUDE.md** how to use the session wrapper for profiling

---

## Conclusion

The document provides a **solid foundation** for separating prefill/decode phases in DeepSeek-OCR profiling. The core architectural understanding is correct, and the code examples are practical. With the recommended additions (validation, troubleshooting, version pinning), this will be a robust reference for profiling workflows.

**Confidence Level**: 🟢 **High** (8.5/10)

**Blocker Issues**: 🟢 **None**

**Recommended Next Steps**:
1. Verify line 403 gate in current code (5 min)
2. Add validation code to decode loop (15 min)
3. Test against one-shot `generate()` for correctness (30 min)
4. Update document with verification section (30 min)

---

**Review Status**: ✅ **APPROVED WITH RECOMMENDATIONS**
