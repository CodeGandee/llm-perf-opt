"""Path utilities.

Helpers to normalize and resolve filesystem paths in a Hydra-aware workflow
without depending directly on Hydra in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_hydra_path(value: Optional[str], cwd: Path) -> Optional[str]:
    """Return an absolute path string for a config-provided path.

    Parameters
    ----------
    value : str or None
        Path value from configuration. May be absolute or relative. ``None`` or
        empty/whitespace-only strings yield ``None``.
    cwd : pathlib.Path
        Base directory to resolve relative paths against (e.g., Hydra runtime cwd).

    Returns
    -------
    str or None
        Absolute path string if input was non-empty; otherwise ``None``.
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

