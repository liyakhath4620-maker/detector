# Prevent local 'packaging' folder from shadowing third-party 'packaging' package in site-packages
import sys
from pathlib import Path

for _p in sys.path:
    if "site-packages" in _p:
        _real_pkg = Path(_p) / "packaging"
        if _real_pkg.is_dir() and str(_real_pkg) not in __path__:
            __path__.append(str(_real_pkg))
            break
