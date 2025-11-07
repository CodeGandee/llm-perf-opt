



# Stakeholder Summary

- Generated: 2025-11-07T03:59:33.609765Z

## Environment

|Key|Value|
| :---: | :---: |
|Device||
|Peak TFLOPs (est.)|209.50|

## Aggregates

|Metric|Mean|Std|
| :---: | :---: | :---: |
|Prefill ms|245.844|723.913|
|Decode ms|628.640|13.558|
|Tokens|64.0|0.0|
|Tokens/s|101.852|2.083|

## Per-Stage Timings (ms)

|Stage|Mean (ms)|Std (ms)|
| :---: | :---: | :---: |
|prefill|245.844|723.913|
|decode|628.640|13.558|
|sam|169.648|653.841|
|clip|10.628|10.761|
|projector|0.047|0.011|


Note: Vision = sam + clip + projector (nested within prefill). Vision ≈ 180.323 ± 664.612 ms; do not add to prefill.
## MFU

|Scope|MFU|
| :---: | :---: |
|Model-level|0.002100|
|Vision|0.000000|
|Prefill|0.008404|
|Decode|0.000238|

## Stage Takeaways

- Decode: Decode dominates runtime (≈ 628.6 ms per run). Tokens/s ≈ 101.85; MFU(decode) ≈ 0.000238.

## Top Operators


No operator records available. See operators.md for details if present.
## Recommendations

- Decode-heavy: consider FlashAttention/fused attention, reduce context, ensure KV cache efficiency.
- Prefill-heavy: check image resolution and preprocessing (base_size/patch_size), consider lighter vision encoder.
- General: verify BF16/FP16 paths, avoid CPU/GPU syncs, repeat runs (≥3) to stabilize metrics.
- Next: Validate with Nsight (Stage 2) for kernel-level attribution and MFU cross-check.
