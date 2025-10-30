"""Hydra-aware workload argv builder.

Build a Python module invocation argv with a set of Hydra overrides. This
utility does not validate the overrides; callers are responsible for providing
valid keys for the target module's config tree.
"""

from __future__ import annotations

from typing import Sequence


def build_work_argv(
    module: str,
    overrides: Sequence[str],
    *,
    hydra_run_dir: str | None = None,
    chdir: bool = True,
    run_mode: str | None = None,
    inputs_manifest: str | None = None,
) -> list[str]:
    """Return a workload argv for ``python -m <module> <overrides...>`` with optional Hydra injections.

    Parameters
    ----------
    module : str
        Python module path to run with ``-m``.
    overrides : sequence of str
        Hydra override strings to append as arguments.
    hydra_run_dir : str or None, optional
        If provided, inject ``hydra.run.dir=<hydra_run_dir>``.
    chdir : bool, default True
        If true, inject ``hydra.job.chdir=true``; otherwise ``false``.
    run_mode : str or None, optional
        If provided, inject ``+run.mode=<run_mode>``.
    inputs_manifest : str or None, optional
        If provided, inject ``+inputs.manifest=<inputs_manifest>``.

    Returns
    -------
    list of str
        Complete argv to run the module with the given overrides.
    """

    args = ["python", "-m", module]
    inj: list[str] = []
    if hydra_run_dir:
        inj.append(f"hydra.run.dir={hydra_run_dir}")
    inj.append(f"hydra.job.chdir={'true' if chdir else 'false'}")
    if run_mode:
        inj.append(f"+run.mode={run_mode}")
    if inputs_manifest:
        inj.append(f"+inputs.manifest={inputs_manifest}")
    return args + list(overrides) + inj
