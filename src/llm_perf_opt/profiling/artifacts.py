"""Artifacts management utilities.

This module provides helpers and a small manager to create and organize the
run artifacts directory tree and write provenance files (env/config/inputs).

Classes
-------
Artifacts
    Manager class for artifacts layout (root/nsys/ncu and pipeline dirs) with read-only property
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
    """Artifacts manager for profiling runs.

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

        # Always resolve to an absolute path to avoid nested/duplicated prefixes
        rp = Path(root).resolve()
        rp.mkdir(parents=True, exist_ok=True)
        # User-facing pipeline output dirs
        (rp / "nsys").mkdir(parents=True, exist_ok=True)
        (rp / "ncu").mkdir(parents=True, exist_ok=True)
        (rp / "static_analysis").mkdir(parents=True, exist_ok=True)
        (rp / "torch_profiler").mkdir(parents=True, exist_ok=True)
        # Ephemeral scratch
        (rp / "tmp").mkdir(parents=True, exist_ok=True)
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

    def out_dir(self, stage: str) -> Path:
        """Return the absolute output directory for a pipeline stage.

        Valid stages: 'static_analysis', 'torch_profiler', 'nsys', 'ncu'.
        The directory is created if missing.
        """

        p = self.root / stage
        p.mkdir(parents=True, exist_ok=True)
        return p

    def tmp_dir(self, stage: str) -> Path:
        """Return the absolute tmp directory for a stage or internal workload.

        Typical stages: 'static_analysis', 'torch_profiler', 'nsys', 'ncu', 'workload'.
        The directory is created if missing.
        """

        p = self.root / "tmp" / stage
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -------------------------------
    # NVTX region path abstractions
    # -------------------------------

    def region_dir(self, region_label: str, create: bool = True) -> Path:
        """Return the filesystem directory for an NVTX region under ``ncu/regions``.

        The directory name is a sanitized variant of the provided NVTX label to
        ensure cross-platform safety. The original label should be preserved in
        JSON/Markdown exports alongside the sanitized path when needed.

        Parameters
        ----------
        region_label : str
            Original NVTX range label (e.g., ``'A'`` or ``'A::A1'``).
        create : bool, default True
            When true, create the directory if it does not exist.

        Returns
        -------
        pathlib.Path
            Absolute path to the region directory under ``ncu/regions``.
        """

        safe = sanitize_region_label(region_label)
        base = self.out_dir("ncu") / "regions" / safe
        if create:
            base.mkdir(parents=True, exist_ok=True)
        return base


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


def sanitize_region_label(label: str) -> str:
    """Return a filesystem-safe folder name for an NVTX region label.

    Rules:
    - Replace path separators and whitespace with ``_``
    - Collapse ``::`` nesting tokens to ``__`` to keep hierarchy readable
    - Keep alphanumerics, ``.``, ``-``, and ``_``; replace others with ``_``

    Examples
    --------
    >>> sanitize_region_label('A::A1')
    'A__A1'
    >>> sanitize_region_label('LLM@decode/all')
    'LLM_decode_all'
    """

    s = str(label).strip()
    # Normalize common separators first
    s = s.replace("::", "__").replace("/", "_").replace("\\", "_")
    # Replace whitespace with underscores
    s = "_".join(s.split())
    # Keep a safe character set
    safe_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    out = []
    for ch in s:
        out.append(ch if ch in safe_chars else "_")
    # Collapse repeated underscores for neatness
    import re as _re  # local import to avoid global dependency

    collapsed = _re.sub(r"_+", "_", "".join(out))
    return collapsed.strip("._") or "region"


def sanitized_region_dir(artifacts: Artifacts, name: str) -> Path:
    """Compatibility helper: return/create ncu/regions/<sanitized(name)> directory."""

    return artifacts.region_dir(name, create=True)
