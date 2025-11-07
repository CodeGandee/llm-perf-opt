



# Stakeholder Summary

- Generated: 2025-11-07T04:00:26.672720Z

## Environment

|Key|Value|
| :---: | :---: |
|Device||
|Peak TFLOPs (est.)|209.50|

## Aggregates

|Metric|Mean|Std|
| :---: | :---: | :---: |
|Prefill ms|246.666|724.383|
|Decode ms|635.403|11.496|
|Tokens|64.0|0.0|
|Tokens/s|100.754|1.708|

## Per-Stage Timings (ms)

|Stage|Mean (ms)|Std (ms)|
| :---: | :---: | :---: |
|prefill|246.666|724.383|
|decode|635.403|11.496|
|sam|28.567|38.892|
|clip|10.538|10.501|
|projector|140.505|612.251|


Note: Vision = sam + clip + projector (nested within prefill). Vision ≈ 179.610 ± 661.643 ms; do not add to prefill.
## MFU

|Scope|MFU|
| :---: | :---: |
|Model-level|0.002087|
|Vision|0.000000|
|Prefill|0.008376|
|Decode|0.000235|

## Stage Takeaways

- Decode: Decode dominates runtime (≈ 635.4 ms per run). Tokens/s ≈ 100.75; MFU(decode) ≈ 0.000235.

## Top Operators


No operator records available. See operators.md for details if present.
## Recommendations

- Decode-heavy: consider FlashAttention/fused attention, reduce context, ensure KV cache efficiency.
- Prefill-heavy: check image resolution and preprocessing (base_size/patch_size), consider lighter vision encoder.
- General: verify BF16/FP16 paths, avoid CPU/GPU syncs, repeat runs (≥3) to stabilize metrics.
- Next: Validate with Nsight (Stage 2) for kernel-level attribution and MFU cross-check.
