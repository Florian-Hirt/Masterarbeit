# Allow importing third_party/safim without packaging it
import sys
from pathlib import Path as _P
_safim = _P(__file__).resolve().parents[2] / "third_party" / "safim"
if _safim.exists() and str(_safim) not in sys.path:
    sys.path.append(str(_safim))
del _P, _safim
