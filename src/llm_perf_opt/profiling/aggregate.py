"""Aggregation helpers for repeated measurements.

Functions
---------
mean_std
    Compute population mean and std for a sequence of floats.
"""

from __future__ import annotations

from statistics import mean, pstdev


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return population mean and std for a list of floats.

    Parameters
    ----------
    values : list[float]
        A list of samples.

    Returns
    -------
    tuple[float, float]
        ``(mean, std)`` where ``std`` is 0.0 if only one sample is given.
    """

    return mean(values), (pstdev(values) if len(values) > 1 else 0.0)
