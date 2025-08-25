#!/usr/bin/env python3
"""
Stillwater KPU Simulator Python API

This module provides a Python interface to the KPU (Knowledge Processing Unit) simulator,
allowing easy testing and educational use of the hardware simulation.
"""

import ctypes
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import os

# Load the shared library
def _load_kpu_library():
    """Load the KPU shared library from common locations."""
    possible_names = [
        "libkpu_simulator.so",      # Linux
        "libkpu_simulator.dylib",   # macOS  
        "kpu_simulator.dll",        # Windows
        "./libkpu_simulator.so",    # Local Linux
        "./libkpu_simulator.dylib", # Local macOS
        "./kpu_simulator.dll"       # Local Windows
    ]
    
    for name in possible_names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    
    raise RuntimeError("Could not load KPU simulator library. Please ensure it's built and in your library path.")

# Global library handle
_kpu_lib = None

def _get_lib():
    global _kpu_lib
    if _kpu_lib is None:
        _kpu_lib = _load_kpu_library()
        _setup_function_signatures()
    return _kpu_lib

# Error codes
class KPUError(Exception):
    """Base exception for KPU simulator errors."""
    pass

class KPUMemoryError(KPUError):
    """Memory access error."""
    pass

class KPUDimensionError(KPUError):
    """Matrix dimension error."""
    pass

# C structure definitions
class MatrixDim(ctypes.Structure):
    _fields_ = [("rows", ctypes.c_uint32),
                ("cols", ctypes.c_uint32)]

def _setup_function_signatures():
    """Setup C function signatures for type safety."""
    lib = _kpu_lib
    
    # Lifecycle functions
    lib.kpu_create.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    lib.kpu_create.restype = ctypes.c_void_p
    
    lib.kpu_destroy.argtypes = [ctypes.c_void_p]
    lib.kpu_destroy.restype = None
    
    # Memory functions
    lib.kpu_main_memory_read.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
    lib.kpu_main_memory_read.restype = ctypes.c_int
    
    lib.kpu_main_memory_write.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
    lib.kpu_main_memory_write.restype = ctypes.c_int
    
    lib.kpu_scratchpad_read.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
    lib.kpu_scratchpad_read.restype = ctypes.c_int
    
    lib.kpu_scratchpad_write.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
    lib.kpu_scratchpad_write.restype = ctypes.c_int
    
    lib.kpu_main_memory_size.argtypes = [ctypes.c_void_p]
    lib.kpu_main_memory_size.restype = ctypes.c_uint64
    
    lib.kpu_scratchpad_size.argtypes = [ctypes.c_void_p]
    lib.kpu_scratchpad_size.restype = ctypes.c_uint64
    
    # DMA functions
    lib.kpu_dma_transfer_sync.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, 
                                         ctypes.c_uint64, ctypes.c_bool, ctypes.c_bool]
    lib.kpu_dma_transfer_sync.restype = ctypes.c_int
    
    # Compute functions
    lib.kpu_matmul_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, 
                                  ctypes.c_uint64, MatrixDim, MatrixDim]
    lib.kpu_matmul_f32.restype = ctypes.c_int
    
    lib.kpu_matmul_accumulate_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, 
                                             ctypes.c_uint64, MatrixDim, MatrixDim]
    lib.kpu_matmul_accumulate_f32.restype = ctypes.c_int
    
    # Utility functions
    lib.kpu_error_string.argtypes = [ctypes.c_int]
    lib.kpu_error_string.restype = ctypes.c_char_p

def _check_error(error_code: int) -> None:
    """Check error code and raise appropriate exception."""
    if error_code == 0:  # KPU_SUCCESS
        return
    
    lib = _get_lib()
    error_msg = lib.kpu_error_string(error_code).decode('utf-8')
    
    if error_code == -2:  # KPU_ERROR_OUT_OF_BOUNDS
        raise KPUMemoryError(error_msg)
    elif error_code == -3:  # KPU_ERROR_INVALID_DIMENSIONS
        raise KPUDimensionError(error_msg)
    else:
        raise KPUError(f"Error {error_code}: {error_msg}")

class KPUSimulator:
    """
    Python interface to the Stillwater KPU Simulator.
    
    This class provides a high-level Python API for the KPU simulator,
    with convenient numpy integration and Pythonic error handling.
    """
    
    def __init__(self, main_memory_size: int = 1<<30, scratchpad_size: int = 1<<20):
        """
        Initialize the KPU simulator.
        
        Args:
            main_memory_size: Size of main memory in bytes (default: 1GB)
            scratchpad_size: Size of scratchpad memory in bytes (default: 1MB)
        """
        lib = _get_lib()
        self._handle = lib.kpu_create(main_memory_size, scratchpad_size)
        if not self._handle:
            raise RuntimeError("Failed to create KPU simulator")
    
    def __del__(self):
        """Cleanup the simulator."""
        if hasattr(self, '_handle') and self._handle:
            lib = _get_lib()
            lib.kpu_destroy(self._handle)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle:
            lib = _get_lib()
            lib.kpu_destroy(self._handle)
            self._handle = None
    
    @property
    def main_memory_size(self) -> int:
        """Get the size of main memory in bytes."""
        lib = _get_lib()
        return lib.kpu_main_memory_size(self._handle)
    
    @property
    def scratchpad_size(self) -> int:
        """Get the size of scratchpad memory in bytes."""
        lib = _get_lib()
        return lib.kpu_scratchpad_size(self._handle)
    
    def write_main_memory(self, addr: int, data: np.ndarray) -> None:
        """
        Write numpy array to main memory.
        
        Args:
            addr: Memory address to write to
            data: Numpy array to write
        """
        lib = _get_lib()
        data_contiguous = np.ascontiguousarray(data)
        size = data_contiguous.nbytes
        
        error = lib.kpu_main_memory_write(
            self._handle, addr, 
            data_contiguous.ctypes.data_as(ctypes.c_void_p), size
        )
        _check_error(error)
    
    def read_main_memory(self, addr: int, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Read numpy array from main memory.
        
        Args:
            addr: Memory address to read from
            dtype: Data type of the array
            shape: Shape of the array
            
        Returns:
            Numpy array containing the data
        """
        lib = _get_lib()
        result = np.zeros(shape, dtype=dtype)
        size = result.nbytes
        
        error = lib.kpu_main_memory_read(
            self._handle, addr,
            result.ctypes.data_as(ctypes.c_void_p), size
        )
        _check_error(error)
        return result
    
    def write_scratchpad(self, addr: int, data: np.ndarray) -> None:
        """
        Write numpy array to scratchpad memory.
        
        Args:
            addr: Memory address to write to
            data: Numpy array to write
        """
        lib = _get_lib()
        data_contiguous = np.ascontiguousarray(data)
        size = data_contiguous.nbytes
        
        error = lib.kpu_scratchpad_write(
            self._handle, addr,
            data_contiguous.ctypes.data_as(ctypes.c_void_p), size
        )
        _check_error(error)
    
    def read_scratchpad(self, addr: int, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Read numpy array from scratchpad memory.
        
        Args:
            addr: Memory address to read from
            dtype: Data type of the array
            shape: Shape of the array
            
        Returns:
            Numpy array containing the data
        """
        lib = _get_lib()
        result = np.zeros(shape, dtype=dtype)
        size = result.nbytes
        
        error = lib.kpu_scratchpad_read(
            self._handle, addr,
            result.ctypes.data_as(ctypes.c_void_p), size
        )
        _check_error(error)
        return result
    
    def dma_transfer(self, src_addr: int, dst_addr: int, size: int,
                    src_main_memory: bool = True, dst_main_memory: bool = False) -> None:
        """
        Perform synchronous DMA transfer.
        
        Args:
            src_addr: Source address
            dst_addr: Destination address  
            size: Number of bytes to transfer
            src_main_memory: True if source is main memory, False if scratchpad
            dst_main_memory: True if destination is main memory, False if scratchpad
        """
        lib = _get_lib()
        error = lib.kpu_dma_transfer_sync(
            self._handle, src_addr, dst_addr, size,
            src_main_memory, dst_main_memory
        )
        _check_error(error)
    
    def matmul(self, A: np.ndarray, B: np.ndarray, 
               addr_A: int = 0, addr_B: Optional[int] = None, addr_C: Optional[int] = None,
               accumulate: bool = False) -> np.ndarray:
        """
        Perform matrix multiplication using the compute engine.
        
        Args:
            A: First matrix (M x K)
            B: Second matrix (K x N)
            addr_A: Scratchpad address for matrix A
            addr_B: Scratchpad address for matrix B (auto-calculated if None)
            addr_C: Scratchpad address for result matrix C (auto-calculated if None)
            accumulate: If True, add result to existing C matrix
            
        Returns:
            Result matrix C (M x N)
        """
        if A.dtype != np.float32 or B.dtype != np.float32:
            raise ValueError("Matrices must be float32")
        
        if len(A.shape) != 2 or len(B.shape) != 2:
            raise ValueError("Matrices must be 2D")
        
        M, K = A.shape
        K2, N = B.shape
        
        if K != K2:
            raise ValueError(f"Matrix dimensions incompatible: A is {M}x{K}, B is {K2}x{N}")
        
        # Calculate addresses if not provided
        if addr_B is None:
            addr_B = addr_A + A.nbytes
        if addr_C is None:
            addr_C = addr_B + B.nbytes
        
        # Write matrices to scratchpad
        self.write_scratchpad(addr_A, A)
        self.write_scratchpad(addr_B, B)
        
        # Initialize result matrix if not accumulating
        if not accumulate:
            C_init = np.zeros((M, N), dtype=np.float32)
            self.write_scratchpad(addr_C, C_init)
        
        # Perform matrix multiplication
        lib = _get_lib()
        dim_A = MatrixDim(M, K)
        dim_B = MatrixDim(K2, N)
        
        if accumulate:
            error = lib.kpu_matmul_accumulate_f32(
                self._handle, addr_A, addr_B, addr_C, dim_A, dim_B
            )
        else:
            error = lib.kpu_matmul_f32(
                self._handle, addr_A, addr_B, addr_C, dim_A, dim_B
            )
        
        _check_error(error)
        
        # Read result
        return self.read_scratchpad(addr_C, np.float32, (M, N))
    
    def benchmark_matmul(self, M: int, N: int, K: int, iterations: int = 100) -> dict:
        """
        Benchmark matrix multiplication performance.
        
        Args:
            M, N, K: Matrix dimensions for C = A @ B where A is MxK and B is KxN
            iterations: Number of iterations for timing
            
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        # Generate random matrices
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # Warm up
        _ = self.matmul(A, B)
        
        # Time KPU matmul
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = self.matmul(A, B)
        kpu_time = (time.perf_counter() - start_time) / iterations
        
        # Time numpy matmul for comparison
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = A @ B
        numpy_time = (time.perf_counter() - start_time) / iterations
        
        # Calculate performance metrics
        ops = 2 * M * N * K  # Multiply-add operations
        kpu_gflops = ops / (kpu_time * 1e9)
        numpy_gflops = ops / (numpy_time * 1e9)
        
        return {
            'matrix_size': f"{M}x{K} @ {K}x{N}",
            'operations': ops,
            'kpu_time_ms': kpu_time * 1000,
            'numpy_time_ms': numpy_time * 1000,
            'kpu_gflops': kpu_gflops,
            'numpy_gflops': numpy_gflops,
            'speedup': numpy_time / kpu_time
        }

# Convenience functions for quick testing
def create_simulator(**kwargs) -> KPUSimulator:
    """Create a KPU simulator with default parameters."""
    return KPUSimulator(**kwargs)

def test_basic_functionality():
    """Run basic functionality tests."""
    print("Testing KPU Simulator basic functionality...")
    
    with create_simulator() as kpu:
        print(f"Main memory size: {kpu.main_memory_size // (1024**3)} GB")
        print(f"Scratchpad size: {kpu.scratchpad_size // (1024**2)} MB")
        
        # Test memory operations
        test_data = np.arange(100, dtype=np.float32)
        kpu.write_main_memory(0, test_data)
        read_data = kpu.read_main_memory(0, np.float32, (100,))
        assert np.allclose(test_data, read_data), "Main memory test failed"
        print("✓ Main memory operations work")
        
        kpu.write_scratchpad(0, test_data)
        read_data = kpu.read_scratchpad(0, np.float32, (100,))
        assert np.allclose(test_data, read_data), "Scratchpad test failed"
        print("✓ Scratchpad operations work")
        
        # Test DMA
        kpu.dma_transfer(0, 0, test_data.nbytes, src_main_memory=True, dst_main_memory=False)
        read_data = kpu.read_scratchpad(0, np.float32, (100,))
        assert np.allclose(test_data, read_data), "DMA test failed"
        print("✓ DMA operations work")
        
        # Test matrix multiplication
        A = np.random.randn(4, 6).astype(np.float32)
        B = np.random.randn(6, 8).astype(np.float32)
        C_kpu = kpu.matmul(A, B)
        C_numpy = A @ B
        assert np.allclose(C_kpu, C_numpy, rtol=1e-5), "Matrix multiplication test failed"
        print("✓ Matrix multiplication works")
        
        print("All tests passed!")

if __name__ == "__main__":
    test_basic_functionality()
            