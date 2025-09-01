# tools/python/stillwater_kpu/__init__.py
"""
Stillwater KPU Simulator Python API

This package provides high-level Python bindings for the Stillwater KPU simulator.
"""

from .simulator import KPUSimulator as Simulator
from .core import *

# Import submodules
from . import core

# Version information
__version__ = "1.0.0"
__author__ = "Stillwater Supercomputing, Inc."
__email__ = "info@stillwater-sc.com"

# Public API
__all__ = [
    "Simulator",
    "core",
    "__version__",
]

def version():
    """Return version string."""
    return __version__

def create_simulator(**kwargs):
    """Convenience function to create a KPU simulator."""
    return Simulator(**kwargs)

