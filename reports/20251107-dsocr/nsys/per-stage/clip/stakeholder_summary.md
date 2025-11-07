



# Stakeholder Summary

- Generated: 2025-11-07T03:59:59.940749Z

## Environment

|Key|Value|
| :---: | :---: |
|Device||
|Peak TFLOPs (est.)|209.50|

## Aggregates

|Metric|Mean|Std|
| :---: | :---: | :---: |
|Prefill ms|240.722|701.981|
|Decode ms|632.276|12.326|
|Tokens|64.0|0.0|
|Tokens/s|101.258|1.889|

## Per-Stage Timings (ms)

|Stage|Mean (ms)|Std (ms)|
| :---: | :---: | :---: |
|prefill|240.722|701.981|
|decode|632.276|12.326|
|sam|28.675|39.271|
|clip|146.075|601.198|
|projector|0.052|0.027|


Note: Vision = sam + clip + projector (nested within prefill). Vision ≈ 174.802 ± 640.496 ms; do not add to prefill.
## MFU

|Scope|MFU|
| :---: | :---: |
|Model-level|0.002114|
|Vision|0.000000|
|Prefill|0.008583|
|Decode|0.000236|

## Stage Takeaways

- Decode: Decode dominates runtime (≈ 632.3 ms per run). Tokens/s ≈ 101.26; MFU(decode) ≈ 0.000236.

## Top Operators


No operator records available. See operators.md for details if present.
## Recommendations

- Decode-heavy: consider FlashAttention/fused attention, reduce context, ensure KV cache efficiency.
- Prefill-heavy: check image resolution and preprocessing (base_size/patch_size), consider lighter vision encoder.
- General: verify BF16/FP16 paths, avoid CPU/GPU syncs, repeat runs (≥3) to stabilize metrics.
- Next: Validate with Nsight (Stage 2) for kernel-level attribution and MFU cross-check.
