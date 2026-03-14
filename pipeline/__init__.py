"""Pipeline assembly utilities for Mansion."""
import sys
from pathlib import Path

# Ensure bundled procthor package is importable (for lights, etc.)
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]  # repo root
_PROCTHOR = _REPO / "procthor"
if _PROCTHOR.exists() and str(_PROCTHOR) not in sys.path:
    sys.path.insert(0, str(_PROCTHOR))

from .graph import build_graph
from .runner import run_pipeline

__all__ = ["build_graph", "run_pipeline"]
