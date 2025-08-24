"""Expose third_party/digraph_transformer as 'digraph_transformer'."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_pkg  = _root / "third_party" / "digraph_transformer"   # <- repo root

# Make 'from digraph_transformer import dataflow_parser' work
__path__ = [str(_pkg)] if _pkg.exists() else []

if _pkg.exists():
    sp = str(_pkg)
    if sp not in sys.path:
        sys.path.insert(0, sp)
