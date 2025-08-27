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

# =============================================================================
# tools/python/stillwater_kpu/core.py  
"""Core functionality and utilities for the KPU simulator."""

import numpy as np
from typing import Dict, Any, List

def get_native_module():
    """Get access to native C++ module for advanced users."""
    try:
        from . import stillwater_kpu_native
        return stillwater_kpu_native
    except ImportError:
        try:
            import stillwater_kpu_native
            return stillwater_kpu_native
        except ImportError:
            return None

def check_system_info() -> Dict[str, Any]:
    """Get system information relevant to KPU simulation."""
    import platform
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total // (1024**3)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 'unknown'
    except ImportError:
        # Fallback if psutil not available
        import os
        memory_gb = 'unknown'
        cpu_count = os.cpu_count()
        cpu_freq = 'unknown'
    
    return {
        'platform': platform.system(),
        'architecture': platform.machine(),
        'python_version': platform.python_version(),
        'memory_gb': memory_gb,
        'cpu_count': cpu_count,
        'cpu_freq_mhz': cpu_freq
    }

def validate_matrix_multiply_shapes(A_shape: tuple, B_shape: tuple) -> tuple:
    """
    Validate matrix multiplication shapes and return result shape.
    
    Returns:
        Result matrix shape (M, N)
        
    Raises:
        ValueError: If shapes are incompatible
    """
    if len(A_shape) != 2 or len(B_shape) != 2:
        raise ValueError("Both arrays must be 2-dimensional")
    
    M, K = A_shape
    K2, N = B_shape
    
    if K != K2:
        raise ValueError(f"Inner dimensions must match: {K} != {K2}")
    
    return (M, N)

def estimate_memory_usage(matrix_shapes: List[tuple], dtype=np.float32) -> int:
    """Estimate memory usage for a list of matrix shapes."""
    bytes_per_element = np.dtype(dtype).itemsize
    total_elements = sum(np.prod(shape) for shape in matrix_shapes)
    return total_elements * bytes_per_element
