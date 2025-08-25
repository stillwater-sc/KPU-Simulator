# =============================================================================
# examples/python/educational/performance_scaling.py
"""
Educational Example: Performance Scaling Analysis

This example demonstrates how KPU performance scales with matrix size
and compares against NumPy.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import stillwater_kpu as kpu

def performance_scaling_analysis():
    """Analyze performance scaling across different matrix sizes."""
    print("📊 Performance Scaling Analysis")
    print("=" * 40)
    
    # Test matrix sizes (square matrices for simplicity)
    sizes = [16, 32, 64, 128, 256, 512]
    iterations = 10  # Fewer iterations for larger matrices
    
    kpu_times = []
    numpy_times = []
    gflops_kpu = []
    gflops_numpy = []
    
    with kpu.Simulator() as sim:
        print(f"🖥️  Using: {sim}")
        print(f"📏 Testing matrix sizes: {sizes}")
        print(f"🔄 Iterations per size: {iterations}")
        print("\nSize\tKPU (ms)\tNumPy (ms)\tKPU GFLOPS\tNumPy GFLOPS\tSpeedup")
        print("-" * 70)
        
        for size in sizes:
            # Generate test matrices
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            
            # Warmup
            _ = sim.matmul(A, B)
            _ = A @ B
            
            # Time KPU
            start_time = time.perf_counter()
            for _ in range(iterations):
                _ = sim.matmul(A, B)
            kpu_time = (time.perf_counter() - start_time) / iterations
            
            # Time NumPy
            start_time = time.perf_counter() 
            for _ in range(iterations):
                _ = A @ B
            numpy_time = (time.perf_counter() - start_time) / iterations
            
            # Calculate GFLOPS (2*N^3 operations for N x N matrices)
            ops = 2 * size**3
            kpu_gf = ops / (kpu_time * 1e9) if kpu_time > 0 else 0
            numpy_gf = ops / (numpy_time * 1e9) if numpy_time > 0 else 0
            speedup = numpy_time / kpu_time if kpu_time > 0 else float('inf')
            
            kpu_times.append(kpu_time * 1000)  # Convert to ms
            numpy_times.append(numpy_time * 1000)
            gflops_kpu.append(kpu_gf)
            gflops_numpy.append(numpy_gf)
            
            print(f"{size}\t{kpu_time*1000:.1f}\t\t{numpy_time*1000:.1f}\t\t"
                  f"{kpu_gf:.1f}\t\t{numpy_gf:.1f}\t\t{speedup:.2f}x")
    
    # Create performance plot
    try:
        plt.figure(figsize=(12, 4))
        
        # Time comparison
        plt.subplot(1, 3, 1)
        plt.loglog(sizes, kpu_times, 'o-', label='KPU', linewidth=2)
        plt.loglog(sizes, numpy_times, 's-', label='NumPy', linewidth=2)
        plt.xlabel('Matrix Size')
        plt.ylabel('Time (ms)')
        plt.title('Computation Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # GFLOPS comparison  
        plt.subplot(1, 3, 2)
        plt.semilogx(sizes, gflops_kpu, 'o-', label='KPU', linewidth=2)
        plt.semilogx(sizes, gflops_numpy, 's-', label='NumPy', linewidth=2)
        plt.xlabel('Matrix Size')
        plt.ylabel('GFLOPS')
        plt.title('Performance (GFLOPS)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Speedup
        plt.subplot(1, 3, 3)
        speedups = [nt/kt for nt, kt in zip(numpy_times, kpu_times)]
        plt.semilogx(sizes, speedups, 'ro-', linewidth=2)
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal Performance')
        plt.xlabel('Matrix Size')
        plt.ylabel('Speedup (NumPy/KPU)')
        plt.title('Relative Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('kpu_performance_scaling.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n📈 Performance plot saved as 'kpu_performance_scaling.png'")
        
    except ImportError:
        print("\n📈 Install matplotlib to see performance plots:")
        print("   pip install matplotlib")

if __name__ == "__main__":
    performance_scaling_analysis()
