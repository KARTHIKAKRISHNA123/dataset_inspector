"""Plugin reader package — readers self-register on import.

Instead of listing every reader module by hand, we auto-import every sibling module in this package
(except `base` and `registry`). Each reader module runs its `@register_reader` decorator at import
time, populating the registry. This is the purest form of the plugin contract: to add a format you
just DROP a new `*_reader.py` file into this folder — no edits anywhere, not even here.
"""

import importlib
import pkgutil

from .base import BaseReader
from .registry import ReaderRegistry, register_reader

# Import all concrete reader modules so their @register_reader decorators run.
_SKIP = {"base", "registry"}
for _m in pkgutil.iter_modules(__path__):
    if _m.name not in _SKIP:
        importlib.import_module(f"{__name__}.{_m.name}")

__all__ = ["BaseReader", "ReaderRegistry", "register_reader"]
