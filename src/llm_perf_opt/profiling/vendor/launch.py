"""Hydra-aware workload argv builder.

Build a Python module invocation argv with a set of Hydra overrides. This
utility does not validate the overrides; callers are responsible for providing
valid keys for the target module's config tree.
"""

from __future__ import annotations

from typing import Sequence


def build_work_argv(module: str, overrides: Sequence[str]) -> list[str]:
    """Return a workload argv for ``python -m <module> <overrides...>``.

    Parameters
    ----------
    module : str
        Python module path to run with ``-m``.
    overrides : sequence of str
        Hydra override strings to append as arguments.

    Returns
    -------
    list of str
        Complete argv to run the module with the given overrides.
    """

    return ["python", "-m", module, *list(overrides)]

