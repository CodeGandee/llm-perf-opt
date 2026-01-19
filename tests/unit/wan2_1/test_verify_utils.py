"""Unit tests for Wan2.1 verification utility helpers."""

from __future__ import annotations

import math

import pytest

from modelmeter.models.wan2_1.scripts.verify._verify_utils import enforce_tolerance, rel_diff, require_finite_non_negative


def test_rel_diff_basic() -> None:
    """Relative difference behaves as expected for typical values."""

    assert rel_diff(analytic=1.0, reference=1.0) == 0.0
    assert rel_diff(analytic=1.1, reference=1.0) == pytest.approx(0.1)
    assert rel_diff(analytic=0.0, reference=1.0) == 1.0


def test_rel_diff_reference_zero() -> None:
    """Relative difference handles the reference==0 corner case."""

    assert rel_diff(analytic=0.0, reference=0.0) == 0.0
    assert math.isinf(rel_diff(analytic=1.0, reference=0.0))


def test_require_finite_non_negative() -> None:
    """Finite/non-negative check rejects NaN/Inf/negative values."""

    require_finite_non_negative("x", 0.0)
    require_finite_non_negative("x", 1.0)
    with pytest.raises(ValueError):
        require_finite_non_negative("x", float("nan"))
    with pytest.raises(ValueError):
        require_finite_non_negative("x", float("inf"))
    with pytest.raises(ValueError):
        require_finite_non_negative("x", -1.0)


def test_enforce_tolerance() -> None:
    """Tolerance enforcement exits when a case exceeds threshold."""

    enforce_tolerance(name="ok", diff=0.01, tolerance=0.05)
    with pytest.raises(SystemExit):
        enforce_tolerance(name="fail", diff=0.2, tolerance=0.05)
