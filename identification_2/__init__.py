"""Import shim so ``python -m identification_2...`` works from repo root.

When executed from inside this repository directory, Python cannot normally
import ``identification_2`` because the package root is also the current
working directory. This shim extends package search paths so root-level
modules/subpackages (``cmaes/``, ``simulator/``, ``models/``) are exposed
under the ``identification_2.*`` namespace.
"""

from __future__ import annotations

from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_repo_root = _pkg_dir.parent

__path__ = [str(_pkg_dir), str(_repo_root)]

