"""
Stub Apple-Silicon-only packages so the test suite runs on Linux CI.

mlx_lm requires Apple Silicon (Metal). On Linux the package is absent, so
any module that does `from mlx_lm import ...` at import time would raise
ImportError before pytest can even collect tests.  We register lightweight
MagicMock stand-ins in sys.modules here, before any test module is imported.
The individual test files still patch the relevant attributes to controlled
return values; this conftest just ensures the import succeeds.
"""
import sys
from unittest.mock import MagicMock

for _mod in ("mlx", "mlx.core", "mlx.core.metal", "mlx_lm"):
    sys.modules.setdefault(_mod, MagicMock())
