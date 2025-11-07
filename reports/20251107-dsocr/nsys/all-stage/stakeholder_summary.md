



# Stakeholder Summary

- Generated: 2025-11-07T04:01:13.555961Z

## Environment

|Key|Value|
| :---: | :---: |
|Device||
|Peak TFLOPs (est.)|209.50|

## Aggregates

|Metric|Mean|Std|
| :---: | :---: | :---: |
|Prefill ms|98.336|64.006|
|Decode ms|707.379|12.346|
|Tokens|64.0|0.0|
|Tokens/s|90.501|1.490|

## Per-Stage Timings (ms)

|Stage|Mean (ms)|Std (ms)|
| :---: | :---: | :---: |
|prefill|98.336|64.006|
|decode|707.379|12.346|
|sam|28.801|40.846|
|clip|11.108|10.849|
|projector|0.056|0.023|


Note: Vision = sam + clip + projector (nested within prefill). Vision ≈ 39.965 ± 51.715 ms; do not add to prefill.
## MFU

|Scope|MFU|
| :---: | :---: |
|Model-level|0.002620|
|Vision|0.000000|
|Prefill|0.021011|
|Decode|0.000211|

## Stage Takeaways

- Decode: Decode dominates runtime (≈ 707.4 ms per run). Tokens/s ≈ 90.50; MFU(decode) ≈ 0.000211.

## Top Operators


No operator records available. See operators.md for details if present.
## Recommendations

- Decode-heavy: consider FlashAttention/fused attention, reduce context, ensure KV cache efficiency.
- Prefill-heavy: check image resolution and preprocessing (base_size/patch_size), consider lighter vision encoder.
- General: verify BF16/FP16 paths, avoid CPU/GPU syncs, repeat runs (≥3) to stabilize metrics.
- Next: Validate with Nsight (Stage 2) for kernel-level attribution and MFU cross-check.
