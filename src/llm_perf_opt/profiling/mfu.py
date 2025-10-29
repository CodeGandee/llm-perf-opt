"""MFU estimation helpers.

Implements simple analytical estimators for decode FLOPs/token and a basic MFU
ratio using achieved throughput vs peak.
"""

from __future__ import annotations

from dataclasses import dataclass


def estimate_decode_flops_per_token(d_model: int, d_ff: int, n_layers: int, ctx_len: int) -> float:
    """Return an approximate FLOPs/token for decode.

    Parameters
    ----------
    d_model : int
        Model hidden size.
    d_ff : int
        Feed-forward dimension.
    n_layers : int
        Number of transformer layers.
    ctx_len : int
        Effective KV/context length.

    Returns
    -------
    float
        Approximate FLOPs per token for decode.
    """

    return float(n_layers) * (4.0 * d_model * d_ff + 2.0 * d_model * d_model + 2.0 * d_model * ctx_len)


def mfu(tokens_per_s: float, flops_per_token: float, peak_tflops: float) -> float:
    """Compute MFU ratio.

    Parameters
    ----------
    tokens_per_s : float
        Achieved token throughput.
    flops_per_token : float
        Analytical compute per token.
    peak_tflops : float
        Theoretical peak TFLOPs for the device/precision.

    Returns
    -------
    float
        MFU ratio in [0, ~]. Values > 1 indicate estimator error or measurement mismatch.
    """

    return (tokens_per_s * flops_per_token) / (peak_tflops * 1e12)


def estimate_prefill_flops_total(d_model: int, d_ff: int, n_layers: int, seq_len: int) -> float:
    """Approximate total FLOPs for prefill of ``seq_len`` tokens.

    Sums the per-token decode-like cost with context length growing from 1..seq_len.
    Closed-form: base_term * seq_len + attn_term * (seq_len * (seq_len + 1) / 2)
    where base_term = n_layers * (4 d_model d_ff + 2 d_model^2) and
          attn_term = n_layers * (2 d_model).
    """

    base = float(n_layers) * (4.0 * d_model * d_ff + 2.0 * d_model * d_model)
    attn = float(n_layers) * (2.0 * d_model)
    return base * float(seq_len) + attn * (float(seq_len) * (float(seq_len) + 1.0) / 2.0)


def select_decode_context_len(
    mode: str, fixed: int | None, prefill_len: int, new_tokens: int, model_window: int | None
) -> int:
    """Select effective decode context length Lctx based on mode.

    - auto: average context length across decode steps: Lp + 0.5*T
    - fixed: use ``fixed``
    - max: use model_window if provided else fall back to Lp + T
    """

    mode_l = (mode or "auto").lower()
    if mode_l == "fixed" and fixed and fixed > 0:
        return int(fixed)
    if mode_l == "max" and model_window and model_window > 0:
        return int(model_window)
    # auto fallback
    return int(max(1, prefill_len + max(0, new_tokens) // 2))


@dataclass
class StageMFU:
    decode: float
    prefill: float
    vision: float
    model_level: float


def compute_stage_mfu(
    prefill_ms: float,
    decode_ms: float,
    vision_ms: float,
    prefill_len: int,
    new_tokens: int,
    d_model: int,
    d_ff: int,
    n_layers: int,
    peak_tflops: float,
    ctx_len_mode: str = "auto",
    ctx_len_fixed: int | None = None,
    model_window: int | None = None,
    vision_flops: float | None = None,
) -> StageMFU:
    """Compute per-stage MFU and an overall model-level MFU."""

    # FLOPs estimates
    Lctx = select_decode_context_len(ctx_len_mode, ctx_len_fixed, int(prefill_len), int(new_tokens), model_window)
    fpt_decode = estimate_decode_flops_per_token(d_model, d_ff, n_layers, Lctx)
    flops_decode = float(max(new_tokens, 0)) * fpt_decode
    flops_prefill = estimate_prefill_flops_total(d_model, d_ff, n_layers, int(prefill_len))
    flops_vision = float(vision_flops or 0.0)

    # Times in seconds
    tp = max(prefill_ms / 1000.0, 1e-9)
    td = max(decode_ms / 1000.0, 1e-9)
    tv = max(vision_ms / 1000.0, 1e-9)

    denom = max(peak_tflops * 1e12, 1e-9)
    m_decode = flops_decode / (denom * td)
    m_prefill = flops_prefill / (denom * tp)
    m_vision = flops_vision / (denom * tv) if flops_vision > 0 else 0.0
    m_model = (flops_decode + flops_prefill + flops_vision) / (denom * (tp + td + tv))
    return StageMFU(decode=m_decode, prefill=m_prefill, vision=m_vision, model_level=m_model)
