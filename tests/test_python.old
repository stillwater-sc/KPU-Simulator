#!/usr/bin/env python3
"""
Test script for the KPU Simulator Python bindings
"""

import numpy as np
import time
import sys

try:
    import stillwater_kpu as kpu
    print(f"✓ Successfully imported stillwater_kpu v{kpu.__version__}")
except ImportError as e:
    print(f"✗ Failed to import stillwater_kpu: {e}")
    print("Make sure the module is built and installed correctly")
    sys.exit(1)

def test_basic_functionality():
    """Test basic simulator functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    # Create simulator configuration
    config = kpu.SimulatorConfig()
    config.external_memory_mb = 128
    config.scratchpad_kb = 64
    config.memory_bandwidth_gbps = 25
    
    # Create simulator
    simulator = kpu.KPUSimulator(config)
    print("✓ Simulator created successfully")
    
    # Test component access
    ext_mem = simulator.get_external_memory()
    scratchpad = simulator.get_scratchpad()
    dma = simulator.get_dma_ext_to_scratch()
    fabric = simulator.get_compute_fabric()
    
    print(f"✓ External memory capacity: {ext_mem.get_capacity() // (1024*1024)} MB")
    print(f"✓ Scratchpad capacity: {scratchpad.get_capacity() // 1024} KB")
    
    # Test reset
    simulator.reset()
    print("✓ Simulator reset successful")
    
    return simulator

def test_matmul_with_python_api():
    """Test matrix multiplication using Python API"""
    print("\n=== Testing Matrix Multiplication (Python API) ===")
    
    # Generate test case
    test = kpu.generate_simple_matmul_test(4, 4, 4)
    print(f"✓ Generated {test.m}x{test.k} * {test.k}x{test.n} test case")
    
    # Create simulator
    simulator = kpu.KPUSimulator()
    
    # Run test
    start_time = time.time()
    success = simulator.run_matmul_test(test)
    end_time = time.time()
    
    if success:
        print("✓ Matrix multiplication test PASSED!")
        print(f"✓ Execution time: {(end_time - start_time)*1000:.2f} ms")
        print(f"✓ Simulation cycles: {simulator.get_current_cycle()}")
        simulator.print_stats()
    else:
        print("✗ Matrix multiplication test FAILED!")
        return False
    
    return True

def test_numpy_integration():
    """Test NumPy array integration"""
    print("\n=== Testing NumPy Integration ===")
    
    # Create test matrices using NumPy
    np.random.seed(42)  # For reproducible results
    A = np.random.randn(6, 8).astype(np.float32)
    B = np.random.randn(8, 5).astype(np.float32)
    
    print(f"✓ Created matrices A({A.shape}) and B({B.shape})")
    
    # Compute expected result using NumPy
    expected_C = A @ B
    print(f"✓ NumPy reference result computed: C{expected_C.shape}")
    
    # Test with KPU simulator
    config = kpu.SimulatorConfig()
    config.external_memory_mb = 256
    config.scratchpad_kb = 128
    
    try:
        start_time = time.time()
        result_C = kpu.numpy_matmul(A, B, config)
        end_time = time.time()
        
        print(f"✓ KPU simulation completed in {(end_time - start_time)*1000:.2f} ms")
        print(f"✓ Result shape: {result_C.shape}")
        
        # Verify results match
        max_diff = np.max(np.abs(result_C - expected_C))
        print(f"✓ Maximum difference from NumPy: {max_diff:.2e}")
        
        if max_diff < 1e-5:
            print("✓ Results match within tolerance!")
            return True
        else:
            print("✗ Results don't match!")
            return False
            
    except Exception as e:
        print(f"✗ NumPy integration test failed: {e}")
        return False

def test_performance_scaling():
    """Test performance with different matrix sizes"""
    print("\n=== Testing Performance Scaling ===")
    
    sizes = [(8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64)]
    
    config = kpu.SimulatorConfig()
    config.external_memory_mb = 512
    config.scratchpad_kb = 256
    
    print("Matrix Size | Sim Time (ms) | Cycles    | Ops/Cycle")
    print("-" * 50)
    
    for m, n, k in sizes:
        A = np.random.randn(m, k).astype(np.float32)
        B = np.random.randn(k, n).astype(np.float32)
        
        try:
            start_time = time.time()
            simulator = kpu.KPUSimulator(config)
            result_C = simulator.run_numpy_matmul(A, B)
            end_time = time.time()
            
            sim_time_ms = (end_time - start_time) * 1000
            cycles = simulator.get_current_cycle()
            ops = 2 * m * n * k  # 2 ops per MAC
            ops_per_cycle = ops / cycles if cycles > 0 else 0
            
            print(f"{m:2}x{n:2}x{k:2}    | {sim_time_ms:8.2f}    | {cycles:8} | {ops_per_cycle:8.2f}")
            
        except Exception as e:
            print(f"{m:2}x{n:2}x{k:2}    | ERROR: {e}")
    
    return True

def test_manual_simulation():
    """Test manual step-by-step simulation"""
    print("\n=== Testing Manual Simulation Control ===")
    
    simulator = kpu.KPUSimulator()
    
    # Create small test matrices
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[2.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    
    print("Manual simulation test:")
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"Expected C = A @ B = \n{A @ B}")
    
    # Reset and get initial state
    simulator.reset()
    print(f"✓ Initial cycle: {simulator.get_current_cycle()}")
    
    # Load data and simulate
    try:
        C = simulator.run_numpy_matmul(A, B)
        print(f"✓ Simulation result:\n{C}")
        print(f"✓ Final cycle: {simulator.get_current_cycle()}")
        
        # Verify correctness
        if np.allclose(C, A @ B, rtol=1e-5):
            print("✓ Manual simulation test PASSED!")
            return True
        else:
            print("✗ Manual simulation test FAILED!")
            return False
            
    except Exception as e:
        print(f"✗ Manual simulation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("KPU Simulator Python Bindings Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Matrix Multiplication API", test_matmul_with_python_api),
        ("NumPy Integration", test_numpy_integration),
        ("Performance Scaling", test_performance_scaling),
        ("Manual Simulation", test_manual_simulation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! KPU Simulator is working correctly.")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())