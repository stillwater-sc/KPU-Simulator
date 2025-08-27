#!/usr/bin/env python3
"""
Demo of KPU core module features - system info, timing, memory planning
"""
import stillwater_kpu as kpu
import numpy as np

def main():
    print("🔧 KPU Core Module Features Demo")
    print("=" * 50)
    
    # System information
    print("\n🖥️  System Information:")
    kpu.core.print_system_info()
    
    # Generate test matrices
    print("\n📊 Matrix Generation and Memory Planning:")
    A, B = kpu.core.generate_test_matrices(100, 200, 150, seed=42)
    print(f"Generated A: {A.shape}, B: {B.shape}")
    
    # Memory estimation
    memory_info = kpu.core.estimate_memory_usage([A.shape, B.shape, (100, 150)])
    print(f"Memory needed: {memory_info['human_readable']}")
    
    # Performance timing
    print("\n⏱️  Performance Timing:")
    with kpu.core.Timer() as timer:
        C_numpy = A @ B
    print(f"NumPy matmul: {timer.elapsed_ms:.2f} ms")
    
    # KPU timing
    with kpu.create_simulator() as sim:
        with kpu.core.Timer() as timer:
            C_kpu = sim.matmul(A, B)
        print(f"KPU matmul: {timer.elapsed_ms:.2f} ms")
        
        # Compare results
        comparison = kpu.core.compare_matrices(C_numpy, C_kpu)
        print(f"Results match: {comparison['matrices_close']}")
        print(f"Max error: {comparison['max_absolute_error']:.2e}")

if __name__ == "__main__":
    main()