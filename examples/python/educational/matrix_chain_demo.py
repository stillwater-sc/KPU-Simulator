# =============================================================================
# examples/python/educational/matrix_chain_demo.py
"""
Educational Example: Matrix Chain Multiplication Optimization

This example demonstrates how different parenthesization of matrix
chain multiplication can affect performance.
"""

import numpy as np
import time
import stillwater_kpu as kpu

def matrix_chain_optimization_demo():
    """Demonstrate matrix chain multiplication optimization."""
    print("⛓️  Matrix Chain Multiplication Optimization")
    print("=" * 55)
    
    # Chain: A @ B @ C where dimensions make order matter
    dim_A = (50, 100)   # 50 x 100
    dim_B = (100, 20)   # 100 x 20  
    dim_C = (20, 80)    # 20 x 80
    
    print(f"Matrix A: {dim_A}")
    print(f"Matrix B: {dim_B}")
    print(f"Matrix C: {dim_C}")
    print(f"Final result: {dim_A[0]} x {dim_C[1]}")
    
    # Calculate theoretical operation counts
    left_ops = dim_A[0] * dim_A[1] * dim_B[1] + dim_A[0] * dim_B[1] * dim_C[1]  # (A@B)@C
    right_ops = dim_B[0] * dim_B[1] * dim_C[1] + dim_A[0] * dim_A[1] * dim_C[1]  # A@(B@C)
    
    print(f"\n🧮 Theoretical Operations:")
    print(f"   Left-to-right (A@B)@C: {left_ops:,}")
    print(f"   Right-to-left A@(B@C): {right_ops:,}")
    print(f"   Theoretical speedup: {left_ops/right_ops:.2f}x")
    
    # Generate test matrices
    np.random.seed(42)
    A = np.random.randn(*dim_A).astype(np.float32)
    B = np.random.randn(*dim_B).astype(np.float32)
    C = np.random.randn(*dim_C).astype(np.float32)
    
    with kpu.Simulator() as sim:
        print(f"\n🖥️  Using: {sim}")
        
        # Method 1: Left-to-right (A @ B) @ C
        print("\n⚡ Left-to-right: (A @ B) @ C")
        start_time = time.perf_counter()
        AB = sim.matmul(A, B)
        result_left = sim.matmul(AB, C)
        left_time = time.perf_counter() - start_time
        
        # Method 2: Right-to-left A @ (B @ C)  
        print("⚡ Right-to-left: A @ (B @ C)")
        start_time = time.perf_counter()
        BC = sim.matmul(B, C)
        result_right = sim.matmul(A, BC)
        right_time = time.perf_counter() - start_time
        
        # Verify results are equivalent
        max_error = np.max(np.abs(result_left - result_right))
        results_match = np.allclose(result_left, result_right, rtol=1e-5)
        
        print(f"\n📈 Performance Results:")
        print(f"   Left-to-right time: {left_time*1000:.2f} ms")
        print(f"   Right-to-left time: {right_time*1000:.2f} ms")
        print(f"   Actual speedup: {left_time/right_time:.2f}x")
        print(f"   Faster method: {'Right-to-left' if right_time < left_time else 'Left-to-right'}")
        print(f"   Max error between methods: {max_error:.2e}")
        print(f"   Results match: {'✅ YES' if results_match else '❌ NO'}")
        
        # Compare with NumPy
        start_time = time.perf_counter()
        numpy_result = A @ B @ C
        numpy_time = time.perf_counter() - start_time
        
        faster_kpu_time = min(left_time, right_time)
        print(f"\n📊 vs NumPy:")
        print(f"   NumPy time: {numpy_time*1000:.2f} ms") 
        print(f"   KPU speedup: {numpy_time/faster_kpu_time:.2f}x")

if __name__ == "__main__":
    matrix_chain_optimization_demo()

