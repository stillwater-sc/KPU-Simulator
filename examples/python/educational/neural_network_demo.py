# examples/python/educational/neural_network_demo.py
"""
Educational Example: Neural Network Layer Computation

This example demonstrates how to use the KPU simulator to compute
a neural network layer: output = input @ weights + bias
"""

import numpy as np
import time
import stillwater_kpu as kpu

def neural_network_layer_demo():
    """Demonstrate neural network layer computation."""
    print("🧠 Neural Network Layer Computation Demo")
    print("=" * 50)
    
    # Network parameters
    batch_size = 32
    input_size = 128  
    output_size = 64
    
    print(f"Batch size: {batch_size}")
    print(f"Input size: {input_size}")
    print(f"Output size: {output_size}")
    
    # Generate synthetic data
    np.random.seed(42)  # For reproducible results
    inputs = np.random.randn(batch_size, input_size).astype(np.float32) * 0.1
    weights = np.random.randn(input_size, output_size).astype(np.float32) * 0.01
    bias = np.random.randn(output_size).astype(np.float32) * 0.01
    
    print(f"Input shape: {inputs.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Bias shape: {bias.shape}")
    
    with kpu.Simulator() as sim:
        print(f"\n🖥️  Using: {sim}")
        
        # Method 1: KPU matrix multiplication + bias
        print("\n⚡ KPU Computation:")
        start_time = time.perf_counter()
        
        # Compute inputs @ weights
        linear_output = sim.matmul(inputs, weights)
        
        # Add bias using broadcasting (convert to matrix operation)
        ones = np.ones((batch_size, 1), dtype=np.float32)
        bias_matrix = bias.reshape(1, -1)  # Shape: (1, output_size)
        bias_broadcast = sim.matmul(ones, bias_matrix)  # Shape: (batch_size, output_size)
        
        # Add bias to linear output
        kpu_output = linear_output + bias_broadcast
        
        kpu_time = time.perf_counter() - start_time
        
        # Method 2: NumPy reference
        print("📊 NumPy Reference:")
        start_time = time.perf_counter()
        numpy_output = inputs @ weights + bias
        numpy_time = time.perf_counter() - start_time
        
        # Compare results
        max_error = np.max(np.abs(kpu_output - numpy_output))
        relative_error = max_error / np.max(np.abs(numpy_output))
        results_match = np.allclose(kpu_output, numpy_output, rtol=1e-5)
        
        print(f"\n📈 Results:")
        print(f"   KPU time: {kpu_time*1000:.2f} ms")
        print(f"   NumPy time: {numpy_time*1000:.2f} ms")
        print(f"   Speedup: {numpy_time/kpu_time:.2f}x")
        print(f"   Max error: {max_error:.2e}")
        print(f"   Relative error: {relative_error:.2e}")
        print(f"   Results match: {'✅ YES' if results_match else '❌ NO'}")
        
        # Show output statistics
        print(f"\n📊 Output Statistics:")
        print(f"   Mean: {np.mean(kpu_output):.4f}")
        print(f"   Std: {np.std(kpu_output):.4f}")
        print(f"   Min: {np.min(kpu_output):.4f}")
        print(f"   Max: {np.max(kpu_output):.4f}")

if __name__ == "__main__":
    neural_network_layer_demo()

