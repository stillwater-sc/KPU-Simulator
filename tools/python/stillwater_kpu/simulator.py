#!/usr/bin/env python3
"""
Stillwater KPU Simulator Python API

This module provides a Python interface to the KPU (Knowledge Processing Unit) simulator,
allowing easy testing and educational use of the hardware simulation.
"""
# tools/python/stillwater_kpu/simulator.py
# Add this mock implementation at the top for development

import numpy as np
from typing import Optional, Tuple, Union, Any
import warnings

# Try to import native module, fall back to mock for development
try:
    try:
        from . import stillwater_kpu_native as _native
    except ImportError:
        import stillwater_kpu_native as _native
    NATIVE_AVAILABLE = True
except ImportError:
    # Mock implementation for development when C++ module isn't built
    warnings.warn(
        "Native C++ module not found. Using mock implementation for development. "
        "Performance will be slow and some features may not work correctly.",
        RuntimeWarning
    )
    _native = None
    NATIVE_AVAILABLE = False

class MockNativeSimulator:
    """Mock implementation for development when C++ module unavailable."""
    
    def __init__(self, main_memory_size: int, scratchpad_size: int):
        self._main_memory_size = main_memory_size
        self._scratchpad_size = scratchpad_size
        self._main_memory = np.zeros(main_memory_size, dtype=np.uint8)
        self._scratchpad = np.zeros(scratchpad_size, dtype=np.uint8)
    
    def main_memory_size(self):
        return self._main_memory_size
    
    def scratchpad_size(self):
        return self._scratchpad_size
    
    def main_memory_write(self, addr: int, data: np.ndarray):
        size = data.nbytes
        if addr + size > len(self._main_memory):
            raise RuntimeError(f"Write out of bounds: {addr + size} > {len(self._main_memory)}")
        self._main_memory[addr:addr + size] = data.view(np.uint8)
    
    def main_memory_read(self, addr: int, dtype: np.dtype, shape: Tuple[int, ...]):
        result = np.zeros(shape, dtype=dtype)
        size = result.nbytes
        if addr + size > len(self._main_memory):
            raise RuntimeError(f"Read out of bounds: {addr + size} > {len(self._main_memory)}")
        result.view(np.uint8).flat[:] = self._main_memory[addr:addr + size]
        return result
    
    def scratchpad_write(self, addr: int, data: np.ndarray):
        size = data.nbytes
        if addr + size > len(self._scratchpad):
            raise RuntimeError(f"Write out of bounds: {addr + size} > {len(self._scratchpad)}")
        self._scratchpad[addr:addr + size] = data.view(np.uint8)
    
    def scratchpad_read(self, addr: int, dtype: np.dtype, shape: Tuple[int, ...]):
        result = np.zeros(shape, dtype=dtype)
        size = result.nbytes
        if addr + size > len(self._scratchpad):
            raise RuntimeError(f"Read out of bounds: {addr + size} > {len(self._scratchpad)}")
        result.view(np.uint8).flat[:] = self._scratchpad[addr:addr + size]
        return result
    
    def dma_transfer_sync(self, src_addr: int, dst_addr: int, size: int, 
                         src_main: bool, dst_main: bool):
        # Simple memcpy simulation
        if src_main and dst_main:
            self._main_memory[dst_addr:dst_addr + size] = self._main_memory[src_addr:src_addr + size]
        elif src_main and not dst_main:
            self._scratchpad[dst_addr:dst_addr + size] = self._main_memory[src_addr:src_addr + size]
        elif not src_main and dst_main:
            self._main_memory[dst_addr:dst_addr + size] = self._scratchpad[src_addr:src_addr + size]
        else:
            self._scratchpad[dst_addr:dst_addr + size] = self._scratchpad[src_addr:src_addr + size]
    
    def matmul_f32(self, addr_A: int, addr_B: int, addr_C: int, 
                   shape_A: tuple, shape_B: tuple, accumulate: bool = False):
        # Read matrices from scratchpad
        A = self.scratchpad_read(addr_A, np.float32, shape_A)
        B = self.scratchpad_read(addr_B, np.float32, shape_B)
        
        # Compute result using NumPy
        if accumulate:
            C = self.scratchpad_read(addr_C, np.float32, (shape_A[0], shape_B[1]))
            result = A @ B + C
        else:
            result = A @ B
        
        # Write result back
        self.scratchpad_write(addr_C, result)

# Mock the native module if not available
if not NATIVE_AVAILABLE:
    class MockNativeModule:
        KPUSimulator = MockNativeSimulator
    
    _native = MockNativeModule()

# Rest of your KPUSimulator class remains the same...
class KPUError(Exception):
    """Base exception for KPU simulator errors.""" 
    pass

class KPUMemoryError(KPUError):
    """Memory access error."""
    pass

class KPUDimensionError(KPUError):
    """Matrix dimension error."""
    pass

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
        if not NATIVE_AVAILABLE:
            print("🐍 Using Python mock implementation (C++ module not available)")
        
        try:
            self._native = _native.KPUSimulator(main_memory_size, scratchpad_size)
        except Exception as e:
            raise RuntimeError(f"Failed to create KPU simulator: {e}")
        
        self._main_memory_size = main_memory_size
        self._scratchpad_size = scratchpad_size
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def __repr__(self) -> str:
        mock_str = " (Mock)" if not NATIVE_AVAILABLE else ""
        return (f"KPUSimulator{mock_str}(main_memory={self._main_memory_size // (1024**3)}GB, "
                f"scratchpad={self._scratchpad_size // (1024**2)}MB)")
    
    @property
    def main_memory_size(self) -> int:
        return self._native.main_memory_size()
    
    @property
    def scratchpad_size(self) -> int:
        return self._native.scratchpad_size()
    
    def write_main_memory(self, addr: int, data: np.ndarray) -> None:
        try:
            data = np.ascontiguousarray(data)
            self._native.main_memory_write(addr, data)
        except Exception as e:
            raise KPUMemoryError(f"Failed to write to main memory at 0x{addr:x}: {e}")
    
    def read_main_memory(self, addr: int, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
        try:
            return self._native.main_memory_read(addr, dtype, shape)
        except Exception as e:
            raise KPUMemoryError(f"Failed to read from main memory at 0x{addr:x}: {e}")
    
    def write_scratchpad(self, addr: int, data: np.ndarray) -> None:
        try:
            data = np.ascontiguousarray(data)
            self._native.scratchpad_write(addr, data)
        except Exception as e:
            raise KPUMemoryError(f"Failed to write to scratchpad at 0x{addr:x}: {e}")
    
    def read_scratchpad(self, addr: int, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
        try:
            return self._native.scratchpad_read(addr, dtype, shape)
        except Exception as e:
            raise KPUMemoryError(f"Failed to read from scratchpad at 0x{addr:x}: {e}")
    
    def dma_transfer(self, src_addr: int, dst_addr: int, size: int,
                    src_main_memory: bool = True, dst_main_memory: bool = False) -> None:
        try:
            self._native.dma_transfer_sync(src_addr, dst_addr, size,
                                         src_main_memory, dst_main_memory)
        except Exception as e:
            raise KPUError(f"DMA transfer failed: {e}")
    
    def matmul(self, A: np.ndarray, B: np.ndarray,
               addr_A: int = 0, addr_B: Optional[int] = None,
               addr_C: Optional[int] = None,
               accumulate: bool = False) -> np.ndarray:
        # Input validation
        if A.dtype != np.float32 or B.dtype != np.float32:
            raise ValueError("Matrices must be float32 dtype")
        
        if len(A.shape) != 2 or len(B.shape) != 2:
            raise ValueError("Matrices must be 2-dimensional")
        
        M, K = A.shape
        K2, N = B.shape
        
        if K != K2:
            raise KPUDimensionError(
                f"Matrix dimensions incompatible: A is {M}x{K}, B is {K2}x{N}"
            )
        
        # Calculate addresses
        if addr_B is None:
            addr_B = addr_A + A.nbytes
        if addr_C is None:
            addr_C = addr_B + B.nbytes
        
        try:
            # Write matrices to scratchpad
            self.write_scratchpad(addr_A, A)
            self.write_scratchpad(addr_B, B)
            
            # Initialize result if not accumulating
            if not accumulate:
                C_init = np.zeros((M, N), dtype=np.float32)
                self.write_scratchpad(addr_C, C_init)
            
            # Perform computation
            self._native.matmul_f32(addr_A, addr_B, addr_C, A.shape, B.shape, accumulate)
            
            # Read result
            return self.read_scratchpad(addr_C, np.float32, (M, N))
            
        except Exception as e:
            raise KPUError(f"Matrix multiplication failed: {e}")
    
    def benchmark_matmul(self, M: int, N: int, K: int, iterations: int = 100) -> dict:
        """Benchmark matrix multiplication performance."""
        import time
        
        # Generate test matrices
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # Warmup
        _ = self.matmul(A, B)
        
        # Time KPU
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = self.matmul(A, B)
        kpu_time = (time.perf_counter() - start_time) / iterations
        
        # Time NumPy reference
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = A @ B
        numpy_time = (time.perf_counter() - start_time) / iterations
        
        # Calculate metrics
        ops = 2 * M * N * K
        kpu_gflops = ops / (kpu_time * 1e9) if kpu_time > 0 else 0
        numpy_gflops = ops / (numpy_time * 1e9) if numpy_time > 0 else 0
        
        return {
            'matrix_size': f"{M}x{K} @ {K}x{N}",
            'operations': ops,
            'kpu_time_ms': kpu_time * 1000,
            'numpy_time_ms': numpy_time * 1000,
            'kpu_gflops': kpu_gflops,
            'numpy_gflops': numpy_gflops,
            'speedup': numpy_time / kpu_time if kpu_time > 0 else float('inf'),
            'using_mock': not NATIVE_AVAILABLE
        }

def create_simulator(**kwargs) -> KPUSimulator:
    """Create a KPU simulator with default parameters."""
    return KPUSimulator(**kwargs)