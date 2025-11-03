"""Manual test for Stage 2 deep profiling.

Runs the Stage 2 runner via the Pixi task `stage2-profile` and asserts that the
expected directory structure and basic provenance files exist under
`tmp/stage2/<run_id>/`.

Note: This script executes profilers which may take time and require an NVIDIA
GPU environment. Keep repeats low and token caps small for quick checks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _latest_run_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    return runs[-1] if runs else None


def main() -> None:
    subprocess.run(["pixi", "run", "stage2-profile"], check=False)
    root = Path("tmp/stage2")
    run_dir = _latest_run_dir(root)
    if run_dir is None:
        raise SystemExit("No Stage 2 run directory found under tmp/stage2")
    assert (run_dir / "nsys").is_dir(), "nsys/ dir missing"
    assert (run_dir / "ncu").is_dir(), "ncu/ dir missing"
    # Provenance
    assert (run_dir / "env.json").is_file(), "env.json missing"
    assert (run_dir / "config.yaml").is_file(), "config.yaml missing"
    assert (run_dir / "inputs.yaml").is_file(), "inputs.yaml missing"
    print("OK: Stage 2 artifacts present in", str(run_dir))


if __name__ == "__main__":
    main()

