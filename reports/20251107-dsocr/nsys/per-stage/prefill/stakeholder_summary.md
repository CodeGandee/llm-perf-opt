



# Stakeholder Summary

- Generated: 2025-11-07T03:58:38.812765Z

## Environment

|Key|Value|
| :---: | :---: |
|Device||
|Peak TFLOPs (est.)|209.50|

## Aggregates

|Metric|Mean|Std|
| :---: | :---: | :---: |
|Prefill ms|249.102|737.597|
|Decode ms|640.785|11.896|
|Tokens|64.0|0.0|
|Tokens/s|99.910|1.731|

## Per-Stage Timings (ms)

|Stage|Mean (ms)|Std (ms)|
| :---: | :---: | :---: |
|prefill|249.102|737.597|
|decode|640.785|11.896|
|sam|28.829|39.812|
|clip|10.665|11.002|
|projector|0.050|0.013|


Note: Vision = sam + clip + projector (nested within prefill). Vision ≈ 39.543 ± 50.827 ms; do not add to prefill.
## MFU

|Scope|MFU|
| :---: | :---: |
|Model-level|0.002384|
|Vision|0.000000|
|Prefill|0.008294|
|Decode|0.000233|

## Stage Takeaways

- Decode: Decode dominates runtime (≈ 640.8 ms per run). Tokens/s ≈ 99.91; MFU(decode) ≈ 0.000233.

## Top Operators


No operator records available. See operators.md for details if present.
## Recommendations

- Decode-heavy: consider FlashAttention/fused attention, reduce context, ensure KV cache efficiency.
- Prefill-heavy: check image resolution and preprocessing (base_size/patch_size), consider lighter vision encoder.
- General: verify BF16/FP16 paths, avoid CPU/GPU syncs, repeat runs (≥3) to stabilize metrics.
- Next: Validate with Nsight (Stage 2) for kernel-level attribution and MFU cross-check.
