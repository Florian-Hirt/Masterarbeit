"""Expose third_party/safim (flat repo) as the package 'safim'."""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
_src  = _root / "third_party" / "safim"
__path__ = [str(_src)] if _src.exists() else []
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
