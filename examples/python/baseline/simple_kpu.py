#!/usr/bin/env python3
"""Simple KPU example script."""

# Clean, simple import
#import stillwater_kpu as kpu
#
# High-level API
#with kpu.Simulator() as sim:
#    result = sim.matmul(A, B)
#
# Access to visualization tools  
#kpu.visualization.plot_performance(data)

import stillwater_kpu as kpu
import numpy as np

def main():
    print("🚀 Simple KPU Simulator Example")
    
    # Create simulator (will use mock if C++ module not built)
    with kpu.Simulator() as sim:
        print(f"📊 {sim}")
        print(f"   Main memory: {sim.main_memory_size // (1024**3)} GB")
        print(f"   Scratchpad: {sim.scratchpad_size // (1024**2)} MB")
        
        # Simple matrix multiplication
        print("\n🧮 Testing matrix multiplication...")
        A = np.random.randn(4, 6).astype(np.float32)
        B = np.random.randn(6, 8).astype(np.float32)
        
        print(f"   A shape: {A.shape}")
        print(f"   B shape: {B.shape}")
        
        # KPU computation
        C_kpu = sim.matmul(A, B)
        
        # Reference computation
        C_numpy = A @ B
        
        # Compare results
        matches = np.allclose(C_kpu, C_numpy, rtol=1e-5)
        max_error = np.max(np.abs(C_kpu - C_numpy))
        
        print(f"   Result shape: {C_kpu.shape}")
        print(f"   Results match: {'✅ YES' if matches else '❌ NO'}")
        print(f"   Max error: {max_error:.2e}")
        
        # Benchmark
        print("\n⏱️  Running benchmark...")
        results = sim.benchmark_matmul(32, 32, 32, iterations=10)
        print(f"   Matrix size: {results['matrix_size']}")
        print(f"   KPU time: {results['kpu_time_ms']:.2f} ms")
        print(f"   NumPy time: {results['numpy_time_ms']:.2f} ms")
        if results.get('using_mock'):
            print("   (Using Python mock - build C++ module for real performance)")

if __name__ == "__main__":
    main()

