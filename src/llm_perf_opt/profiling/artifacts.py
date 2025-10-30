from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf  # type: ignore[import-untyped]


def new_run_id(dt: datetime | None = None) -> str:
    """Return a timestamped run identifier: YYYYMMDD-HHMMSS.

    Parameters
    ----------
    dt : datetime | None
        Optional datetime; defaults to ``datetime.now()`` if not provided.
    """

    ts = (dt or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return ts


@dataclass
class Artifacts:
    """Artifacts manager for Stage 2 runs.

    Creates the basic layout under ``root`` and provides path helpers.
    """

    root: Path

    def __post_init__(self) -> None:  # pragma: no cover - simple FS initializer
        (self.root).mkdir(parents=True, exist_ok=True)
        (self.root / "nsys").mkdir(parents=True, exist_ok=True)
        (self.root / "ncu").mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / name


def create_stage2_root(base_dir: Path | str = "tmp/stage2") -> Path:
    """Create and return a new Stage 2 artifacts root: tmp/stage2/<run_id>/.

    Parameters
    ----------
    base_dir : Path | str, default 'tmp/stage2'
        Base directory for Stage 2 artifacts.
    """

    base = Path(base_dir)
    rid = new_run_id()
    root = base / rid
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_env_json(path: Path) -> None:
    """Write environment snapshot to JSON at ``path``.

    Wrapper around the Stage 1 helper so Stage 2 code can use a local import.
    """

    from llm_perf_opt.profiling.hw import write_env_json as _write_env

    _write_env(str(path))


def write_config_yaml(path: Path, cfg: Any) -> None:
    """Serialize a Hydra/OmegaConf config object to YAML at ``path``."""

    yml = OmegaConf.to_yaml(cfg)
    path.write_text(yml, encoding="utf-8")


def write_inputs_yaml(path: Path, records: list[dict]) -> None:
    """Write a minimal inputs manifest to YAML at ``path``.

    Parameters
    ----------
    records : list[dict]
        A list of dictionaries with at least a ``path`` key. Optional fields
        may include ``bytes``, ``width``, ``height``.
    """

    payload = {"count": len(records), "inputs": records}
    yml = OmegaConf.to_yaml(payload)
    path.write_text(yml, encoding="utf-8")

