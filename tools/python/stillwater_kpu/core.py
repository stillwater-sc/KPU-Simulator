# tools/python/stillwater_kpu/core.py
"""
Core functionality and utilities for the Stillwater KPU simulator.

This module provides essential utilities, system information, validation functions,
and helper classes that are used throughout the KPU simulator.
"""

import numpy as np
import platform
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

# =============================================================================
# System Information and Diagnostics
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information relevant to KPU simulation.
    
    Returns:
        Dictionary with system specifications and capabilities
    """
    info = {
        'platform': platform.system(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
    }
    
    # Try to get more detailed system info
    try:
        import psutil
        memory = psutil.virtual_memory()
        info.update({
            'memory_total_gb': memory.total // (1024**3),
            'memory_available_gb': memory.available // (1024**3), 
            'memory_percent_used': memory.percent,
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
        })
        
        # CPU frequency if available
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            info['cpu_freq_current_mhz'] = cpu_freq.current
            info['cpu_freq_max_mhz'] = cpu_freq.max
            
    except ImportError:
        # Fallback without psutil
        import os
        info.update({
            'memory_total_gb': 'unknown (install psutil for details)',
            'cpu_count_logical': os.cpu_count(),
            'cpu_count_physical': 'unknown',
        })
    
    # Check for accelerator libraries
    info['accelerators'] = check_accelerator_support()
    
    return info

def check_accelerator_support() -> Dict[str, bool]:
    """Check what acceleration libraries are available."""
    support = {}
    
    # OpenMP (via numpy)
    try:
        import numpy as np
        # Check if numpy was built with OpenMP
        config = np.__config__.show()
        support['numpy_openmp'] = 'openblas' in str(config).lower() or 'mkl' in str(config).lower()
    except:
        support['numpy_openmp'] = False
    
    # CUDA
    try:
        import torch
        support['cuda'] = torch.cuda.is_available()
        if support['cuda']:
            support['cuda_devices'] = torch.cuda.device_count()
            support['cuda_version'] = torch.version.cuda
    except ImportError:
        support['cuda'] = False
    
    # Intel MKL
    try:
        import numpy as np
        support['intel_mkl'] = 'mkl' in np.__config__.show().lower()
    except:
        support['intel_mkl'] = False
    
    return support

def print_system_info():
    """Print formatted system information."""
    info = get_system_info()
    
    print("🖥️  System Information:")
    print(f"   Platform: {info['platform']} {info['architecture']}")
    print(f"   Processor: {info.get('processor', 'Unknown')}")
    print(f"   Python: {info['python_implementation']} {info['python_version']}")
    
    if 'memory_total_gb' in info and isinstance(info['memory_total_gb'], int):
        print(f"   Memory: {info['memory_available_gb']}/{info['memory_total_gb']} GB available ({info['memory_percent_used']:.1f}% used)")
    
    if 'cpu_count_logical' in info:
        logical = info['cpu_count_logical'] 
        physical = info.get('cpu_count_physical', 'unknown')
        print(f"   CPUs: {logical} logical, {physical} physical")
    
    if 'cpu_freq_current_mhz' in info:
        print(f"   CPU Frequency: {info['cpu_freq_current_mhz']:.0f} MHz (max: {info['cpu_freq_max_mhz']:.0f} MHz)")
    
    print("🚀 Accelerator Support:")
    for name, available in info['accelerators'].items():
        status = "✅" if available else "❌"
        print(f"   {name}: {status}")

# =============================================================================
# Matrix and Memory Utilities
# =============================================================================

def validate_matrix_shapes(A_shape: Tuple[int, ...], B_shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    """
    Validate matrix multiplication shapes and return M, K, N dimensions.
    
    Args:
        A_shape: Shape of matrix A
        B_shape: Shape of matrix B
        
    Returns:
        Tuple of (M, K, N) where result is M x N
        
    Raises:
        ValueError: If shapes are invalid or incompatible
    """
    if len(A_shape) != 2 or len(B_shape) != 2:
        raise ValueError("Both matrices must be 2-dimensional")
    
    M, K = A_shape
    K2, N = B_shape
    
    if K != K2:
        raise ValueError(f"Inner dimensions must match for matrix multiplication: {K} != {K2}")
    
    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError("All matrix dimensions must be positive")
    
    return M, K, N

def estimate_memory_usage(shapes: List[Tuple[int, ...]], dtype: np.dtype = np.float32) -> Dict[str, Union[int, str]]:
    """
    Estimate memory usage for a list of matrix shapes.
    
    Args:
        shapes: List of matrix shapes
        dtype: Data type for calculations
        
    Returns:
        Dictionary with memory estimates in various units
    """
    bytes_per_element = np.dtype(dtype).itemsize
    total_elements = sum(np.prod(shape) for shape in shapes)
    total_bytes = total_elements * bytes_per_element
    
    return {
        'elements': total_elements,
        'bytes': total_bytes,
        'kb': total_bytes / 1024,
        'mb': total_bytes / (1024**2),
        'gb': total_bytes / (1024**3),
        'dtype': str(dtype),
        'bytes_per_element': bytes_per_element,
        'human_readable': format_bytes(total_bytes)
    }

def format_bytes(bytes: int) -> str:
    """Format bytes in human readable form."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} PB"

def generate_test_matrices(M: int, K: int, N: int, 
                         dtype: np.dtype = np.float32,
                         seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate reproducible test matrices for benchmarking.
    
    Args:
        M, K, N: Matrix dimensions (A: M×K, B: K×N)
        dtype: NumPy data type
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (A, B) matrices
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate matrices with reasonable ranges for numerical stability
    if dtype == np.float32:
        # Float32: use smaller range to avoid overflow in large multiplications
        A = np.random.uniform(-0.5, 0.5, (M, K)).astype(dtype)
        B = np.random.uniform(-0.5, 0.5, (K, N)).astype(dtype)
    elif dtype == np.float64:
        A = np.random.uniform(-1.0, 1.0, (M, K)).astype(dtype)
        B = np.random.uniform(-1.0, 1.0, (K, N)).astype(dtype)
    elif dtype in [np.int32, np.int64]:
        A = np.random.randint(-10, 10, (M, K)).astype(dtype)
        B = np.random.randint(-10, 10, (K, N)).astype(dtype)
    else:
        # Default: normal distribution
        A = np.random.randn(M, K).astype(dtype)
        B = np.random.randn(K, N).astype(dtype)
    
    return A, B

# =============================================================================
# Performance Utilities
# =============================================================================

class Timer:
    """High-resolution timer for performance measurements."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
    
    def start(self):
        """Start the timer."""
        self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop the timer."""
        self.end_time = time.perf_counter()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000
    
    @property
    def elapsed_us(self) -> float:
        """Get elapsed time in microseconds."""
        return self.elapsed * 1_000_000

def benchmark_function(func, *args, iterations: int = 100, warmup: int = 5, **kwargs) -> Dict[str, float]:
    """
    Benchmark a function with multiple iterations.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        iterations: Number of timing iterations
        warmup: Number of warmup iterations
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Actual timing
    times = []
    for _ in range(iterations):
        with Timer() as timer:
            result = func(*args, **kwargs)
        times.append(timer.elapsed)
    
    times = np.array(times)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'median_ms': np.median(times) * 1000,
        'iterations': iterations,
        'warmup': warmup,
        'total_time_s': np.sum(times)
    }

def calculate_gflops(M: int, K: int, N: int, time_seconds: float) -> float:
    """
    Calculate GFLOPS for matrix multiplication.
    
    Args:
        M, K, N: Matrix dimensions
        time_seconds: Execution time in seconds
        
    Returns:
        GFLOPS (billion floating-point operations per second)
    """
    # Matrix multiplication: 2*M*K*N operations (multiply + add for each element)
    ops = 2 * M * K * N
    return ops / (time_seconds * 1e9)

# =============================================================================
# Configuration and Validation
# =============================================================================

def get_optimal_tile_size(available_memory_bytes: int, 
                         element_size: int = 4,
                         safety_factor: float = 0.8) -> int:
    """
    Calculate optimal tile size for blocked matrix operations.
    
    Args:
        available_memory_bytes: Available memory for computation
        element_size: Size of each matrix element in bytes
        safety_factor: Safety factor to avoid memory overflow
        
    Returns:
        Suggested tile size (square tiles)
    """
    # For three square matrices (A, B, C) of size N×N
    # Memory needed: 3 * N^2 * element_size
    usable_memory = available_memory_bytes * safety_factor
    max_elements_per_matrix = usable_memory / (3 * element_size)
    tile_size = int(np.sqrt(max_elements_per_matrix))
    
    # Round down to nearest multiple of 8 for better alignment
    tile_size = (tile_size // 8) * 8
    
    return max(8, tile_size)  # Minimum tile size of 8

def validate_scratchpad_capacity(shapes: List[Tuple[int, ...]], 
                               available_bytes: int,
                               dtype: np.dtype = np.float32) -> Dict[str, Any]:
    """
    Validate if matrices fit in scratchpad memory.
    
    Args:
        shapes: List of matrix shapes to store
        available_bytes: Available scratchpad memory
        dtype: Data type
        
    Returns:
        Dictionary with validation results
    """
    memory_info = estimate_memory_usage(shapes, dtype)
    required_bytes = memory_info['bytes']
    
    fits = required_bytes <= available_bytes
    utilization = required_bytes / available_bytes if available_bytes > 0 else float('inf')
    
    return {
        'fits': fits,
        'required_bytes': required_bytes,
        'available_bytes': available_bytes,
        'utilization_percent': utilization * 100,
        'memory_info': memory_info,
        'recommendation': 'OK' if fits else 'Consider smaller matrices or blocked computation'
    }

# =============================================================================
# Debugging and Diagnostics
# =============================================================================

def compare_matrices(A: np.ndarray, B: np.ndarray, 
                    rtol: float = 1e-5, atol: float = 1e-8) -> Dict[str, Any]:
    """
    Compare two matrices and provide detailed difference analysis.
    
    Args:
        A, B: Matrices to compare
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Dictionary with comparison results
    """
    if A.shape != B.shape:
        return {
            'shapes_match': False,
            'A_shape': A.shape,
            'B_shape': B.shape,
            'error': 'Shape mismatch'
        }
    
    diff = np.abs(A - B)
    rel_diff = diff / (np.abs(A) + 1e-15)  # Avoid division by zero
    
    close = np.allclose(A, B, rtol=rtol, atol=atol)
    
    return {
        'shapes_match': True,
        'matrices_close': close,
        'max_absolute_error': np.max(diff),
        'mean_absolute_error': np.mean(diff),
        'max_relative_error': np.max(rel_diff),
        'mean_relative_error': np.mean(rel_diff),
        'fraction_elements_close': np.mean(np.isclose(A, B, rtol=rtol, atol=atol)),
        'tolerance_used': {'rtol': rtol, 'atol': atol},
        'A_stats': {'min': np.min(A), 'max': np.max(A), 'mean': np.mean(A), 'std': np.std(A)},
        'B_stats': {'min': np.min(B), 'max': np.max(B), 'mean': np.mean(B), 'std': np.std(B)}
    }

# =============================================================================
# Module Information
# =============================================================================

def get_native_module():
    """
    Attempt to get access to native C++ module for advanced users.
    
    Returns:
        Native module if available, None otherwise
    """
    try:
        from . import stillwater_kpu_native
        return stillwater_kpu_native
    except ImportError:
        try:
            import stillwater_kpu_native
            return stillwater_kpu_native
        except ImportError:
            return None

def is_native_available() -> bool:
    """Check if native C++ module is available."""
    return get_native_module() is not None

# Export commonly used functions
__all__ = [
    # System info
    'get_system_info', 'print_system_info', 'check_accelerator_support',
    
    # Matrix utilities
    'validate_matrix_shapes', 'estimate_memory_usage', 'generate_test_matrices',
    'format_bytes',
    
    # Performance
    'Timer', 'benchmark_function', 'calculate_gflops',
    
    # Configuration
    'get_optimal_tile_size', 'validate_scratchpad_capacity',
    
    # Debugging
    'compare_matrices',
    
    # Module info
    'get_native_module', 'is_native_available'
]