"""Expose third_party/python_graphs/python_graphs as 'python_graphs'."""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
_pkg  = _root / "third_party" / "python_graphs" / "python_graphs"
__path__ = [str(_pkg)] if _pkg.exists() else []
if _pkg.exists():
    sp = str(_pkg)
    if sp not in sys.path:
        sys.path.insert(0, sp)
