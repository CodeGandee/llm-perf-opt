"""Stage 2 artifacts management utilities.

This module provides helpers and a small manager to create and organize the
Stage 2 artifacts directory tree and write provenance files (env/config/inputs).

Classes
-------
Artifacts
    Manager class for artifacts layout (root/nsys/ncu) with read-only property
    access and explicit setters/factories per coding guidelines.

Functions
---------
new_run_id
    Build a timestamp-based run identifier (YYYYMMDD-HHMMSS).
create_stage2_root
    Create `tmp/stage2/<run_id>/` and return the `Path`.
write_env_json
    Persist environment snapshot to a JSON file.
write_config_yaml
    Serialize a Hydra/OmegaConf config to YAML.
write_inputs_yaml
    Write a minimal inputs manifest (count + list of records) to YAML.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from omegaconf import OmegaConf  # type: ignore[import-untyped]

T = TypeVar("T", bound="Artifacts")


def new_run_id(dt: Optional[datetime] = None) -> str:
    """Return a timestamped run identifier.

    Parameters
    ----------
    dt : datetime or None, optional
        Datetime to format; defaults to ``datetime.now()``.

    Returns
    -------
    str
        Identifier in the form ``YYYYMMDD-HHMMSS``.

    Examples
    --------
    >>> rid = new_run_id()
    >>> len(rid) == 15
    True
    """

    ts = (dt or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return ts


class Artifacts:
    """Artifacts manager for Stage 2 runs.

    This class follows the project OO guidelines: constructor takes no
    arguments; use the `from_root()` factory or the `set_root()` mutator to
    configure the target directory. Member variables are prefixed with `m_` and
    read-only access is provided via properties.

    Attributes
    ----------
    root : pathlib.Path
        Read-only property for the artifacts root directory.
    """

    def __init__(self) -> None:
        self.m_root: Optional[Path] = None

    @property
    def root(self) -> Path:
        """Artifacts root directory (read-only)."""

        if self.m_root is None:
            raise RuntimeError("Artifacts root not set. Use from_root() or set_root().")
        return self.m_root

    def set_root(self, root: Path | str) -> None:
        """Set and prepare the artifacts root directory.

        Parameters
        ----------
        root : Path or str
            Destination root for this artifacts manager.
        """

        rp = Path(root)
        rp.mkdir(parents=True, exist_ok=True)
        (rp / "nsys").mkdir(parents=True, exist_ok=True)
        (rp / "ncu").mkdir(parents=True, exist_ok=True)
        self.m_root = rp

    @classmethod
    def from_root(cls: Type[T], root: Path | str) -> T:
        """Factory that returns an initialized manager for ``root``.

        Examples
        --------
        >>> a = Artifacts.from_root('tmp/stage2/demo')
        >>> a.root.name == 'demo'
        True
        """

        obj = cls()
        obj.set_root(root)
        return obj

    def path(self, name: str) -> Path:
        """Return a path within the artifacts root.

        Parameters
        ----------
        name : str
            File name or relative subpath under the root.
        """

        return self.root / name


def create_stage2_root(base_dir: Path | str = "tmp/stage2") -> Path:
    """Create and return a new Stage 2 artifacts root.

    The directory layout is ``tmp/stage2/<run_id>/``.

    Parameters
    ----------
    base_dir : Path or str, optional
        Base directory for Stage 2 artifacts, by default ``'tmp/stage2'``.

    Returns
    -------
    Path
        The created artifacts root directory.
    """

    base = Path(base_dir)
    rid = new_run_id()
    root = base / rid
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_env_json(path: Path) -> None:
    """Write environment snapshot to JSON at ``path``.

    Notes
    -----
    Wrapper around the Stage 1 helper so Stage 2 code can use a local import.
    """

    from llm_perf_opt.profiling.hw import write_env_json as _write_env

    _write_env(str(path))


def write_config_yaml(path: Path, cfg: Any) -> None:
    """Serialize a Hydra/OmegaConf config object to YAML at ``path``.

    Parameters
    ----------
    path : Path
        Destination file path.
    cfg : Any
        Hydra/OmegaConf configuration object (or any OmegaConf‑serializable object).
    """

    yml = OmegaConf.to_yaml(cfg)
    path.write_text(yml, encoding="utf-8")


def write_inputs_yaml(path: Path, records: list[dict]) -> None:
    """Write a minimal inputs manifest to YAML at ``path``.

    Parameters
    ----------
    path : Path
        Destination file path.
    records : list of dict
        A list of dictionaries with at least a ``path`` key. Optional fields
        may include ``bytes``, ``width``, ``height``.
    """

    payload = {"count": len(records), "inputs": records}
    yml = OmegaConf.to_yaml(payload)
    path.write_text(yml, encoding="utf-8")
