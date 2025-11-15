"""
Top-level WDR Python package.
"""

from importlib import import_module

# Expose the compiled coder module as wdr.coder
coder = import_module(".coder", __name__)

# Re-export utils for convenience
from . import utils

__all__ = ["coder", "utils"]
