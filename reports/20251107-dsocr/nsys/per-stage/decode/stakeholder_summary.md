



# Stakeholder Summary

- Generated: 2025-11-07T03:59:06.890158Z

## Environment

|Key|Value|
| :---: | :---: |
|Device||
|Peak TFLOPs (est.)|209.50|

## Aggregates

|Metric|Mean|Std|
| :---: | :---: | :---: |
|Prefill ms|93.869|60.816|
|Decode ms|814.783|764.727|
|Tokens|64.0|0.0|
|Tokens/s|95.871|18.461|

## Per-Stage Timings (ms)

|Stage|Mean (ms)|Std (ms)|
| :---: | :---: | :---: |
|prefill|93.869|60.816|
|decode|814.783|764.727|
|sam|28.726|39.648|
|clip|10.549|10.204|
|projector|0.048|0.009|


Note: Vision = sam + clip + projector (nested within prefill). Vision ≈ 39.323 ± 49.858 ms; do not add to prefill.
## MFU

|Scope|MFU|
| :---: | :---: |
|Model-level|0.002337|
|Vision|0.000000|
|Prefill|0.022011|
|Decode|0.000183|

## Stage Takeaways

- Decode: Decode dominates runtime (≈ 814.8 ms per run). Tokens/s ≈ 95.87; MFU(decode) ≈ 0.000183.

## Top Operators


No operator records available. See operators.md for details if present.
## Recommendations

- Decode-heavy: consider FlashAttention/fused attention, reduce context, ensure KV cache efficiency.
- Prefill-heavy: check image resolution and preprocessing (base_size/patch_size), consider lighter vision encoder.
- General: verify BF16/FP16 paths, avoid CPU/GPU syncs, repeat runs (≥3) to stabilize metrics.
- Next: Validate with Nsight (Stage 2) for kernel-level attribution and MFU cross-check.
