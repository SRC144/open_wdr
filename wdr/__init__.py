"""
Top-level WDR Python package.
"""

from importlib import import_module

# Expose the compiled coder module as wdr.coder
coder = import_module(".coder", __name__)

# Re-export utils for convenience
from . import utils

# Re-export the container module (File Format Logic)
from . import container

# Re-export the io module (File Format Logic)
from . import io

__all__ = ["coder", "utils", "container", "io"]