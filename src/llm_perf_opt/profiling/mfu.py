"""MFU estimation helpers.

Implements simple analytical estimators for decode FLOPs/token and a basic MFU
ratio using achieved throughput vs peak.
"""

from __future__ import annotations

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
