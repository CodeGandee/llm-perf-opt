"""Path utilities.

Helpers to normalize and resolve filesystem paths in a Hydra-aware workflow
without depending directly on Hydra in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_hydra_path(value: Optional[str], cwd: Path) -> Optional[str]:
    """
    Return an absolute path string for a config-provided path.

    Parameters
    ----------
    value : str or None
        Path value from configuration. May be absolute or relative. ``None`` or
        empty/whitespace-only strings yield ``None``.
    cwd : pathlib.Path
        Base directory to resolve relative paths against (for example,
        the Hydra runtime working directory).

    Returns
    -------
    str or None
        Absolute path string if the input was non-empty; otherwise ``None``.
    """

    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "null":
        return None
    p = Path(s)
    if p.is_absolute():
        return str(p.resolve())
    base = cwd if isinstance(cwd, Path) else Path(str(cwd))
    return str((base / p).resolve())


def workspace_root() -> str:
    """
    Return the absolute path to the workspace root.

    The root is inferred by walking parents of this file until a directory
    containing ``pyproject.toml`` or ``.git`` is found.

    Returns
    -------
    str
        Absolute path to the detected workspace root directory.
    """

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() or (parent / ".git").is_dir():
            return str(parent)
    # Fallback: top-most parent from ``here.parents``.
    return str(here.parents[-1])


def analytic_model_dir(run_id: str) -> str:
    """
    Return the directory for analytic artifacts for a given run.

    The output path is always absolute and rooted under::

        tmp/profile-output/<run_id>/static_analysis/analytic_model

    Parameters
    ----------
    run_id : str
        Identifier for the profiling or analytic run.

    Returns
    -------
    str
        Absolute path to the analytic model directory for the given run.
    """

    root = Path(workspace_root())
    return str(root / "tmp" / "profile-output" / run_id / "static_analysis" / "analytic_model")


def analytic_layer_docs_dir(run_id: str) -> str:
    """
    Return the directory for per-layer Markdown docs.

    The path is always absolute and rooted under the analytic model directory.

    Parameters
    ----------
    run_id : str
        Identifier for the profiling or analytic run.

    Returns
    -------
    str
        Absolute path to the directory where per-layer Markdown docs
        should be written.
    """

    base = Path(analytic_model_dir(run_id))
    return str(base / "layers")
