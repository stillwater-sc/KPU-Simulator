#!/usr/bin/env python3
"""
Test script for the TopLevelSimulator Python bindings
"""

import sys
import traceback

def test_basic_import():
    """Test that we can import the TopLevelSimulator module"""
    print("=== Testing Basic Import ===")
    try:
        import stillwater_toplevel as tl
        print(f"✓ Successfully imported stillwater_toplevel v{getattr(tl, '__version__', 'unknown')}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import stillwater_toplevel: {e}")
        return False

def test_basic_functionality():
    """Test basic simulator lifecycle"""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        import stillwater_toplevel as tl
        
        # Create simulator
        simulator = tl.TopLevelSimulator()
        print("✓ TopLevelSimulator created successfully")
        
        # Test initial state
        if not simulator.is_initialized():
            print("✓ Simulator starts uninitialized")
        else:
            print("✗ Simulator should start uninitialized")
            return False
            
        # Initialize
        if simulator.initialize():
            print("✓ Simulator initialization successful")
        else:
            print("✗ Simulator initialization failed")
            return False
            
        # Check initialized state
        if simulator.is_initialized():
            print("✓ Simulator reports initialized state correctly")
        else:
            print("✗ Simulator should be initialized")
            return False
            
        # Run self test
        if simulator.run_self_test():
            print("✓ Self test passed")
        else:
            print("✗ Self test failed")
            return False
            
        # Shutdown
        simulator.shutdown()
        if not simulator.is_initialized():
            print("✓ Simulator shutdown successfully")
        else:
            print("✗ Simulator should be uninitialized after shutdown")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_context_manager():
    """Test context manager functionality"""
    print("\n=== Testing Context Manager ===")
    
    try:
        import stillwater_toplevel as tl
        
        with tl.TopLevelSimulator() as simulator:
            print("✓ Context manager entry successful")
            
            if simulator.is_initialized():
                print("✓ Simulator auto-initialized in context")
            else:
                print("✗ Simulator should be initialized in context")
                return False
                
            if simulator.run_self_test():
                print("✓ Self test passed in context")
            else:
                print("✗ Self test failed in context")
                return False
                
        print("✓ Context manager exit successful")
        return True
        
    except Exception as e:
        print(f"✗ Context manager test failed: {e}")
        traceback.print_exc()
        return False

def test_error_conditions():
    """Test error conditions"""
    print("\n=== Testing Error Conditions ===")
    
    try:
        import stillwater_toplevel as tl
        
        simulator = tl.TopLevelSimulator()
        
        # Self test should fail when not initialized
        if not simulator.run_self_test():
            print("✓ Self test correctly fails when not initialized")
        else:
            print("✗ Self test should fail when not initialized")
            return False
            
        # Multiple initializations should be safe
        simulator.initialize()
        first_init_state = simulator.is_initialized()
        simulator.initialize()  # Second call
        second_init_state = simulator.is_initialized()
        
        if first_init_state and second_init_state:
            print("✓ Multiple initializations handled correctly")
        else:
            print("✗ Multiple initializations not handled correctly")
            return False
            
        # Multiple shutdowns should be safe
        simulator.shutdown()
        first_shutdown_state = not simulator.is_initialized()
        simulator.shutdown()  # Second call
        second_shutdown_state = not simulator.is_initialized()
        
        if first_shutdown_state and second_shutdown_state:
            print("✓ Multiple shutdowns handled correctly")
        else:
            print("✗ Multiple shutdowns not handled correctly")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error conditions test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("TopLevelSimulator Python Bindings Test Suite")
    print("=" * 70)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("Basic Functionality", test_basic_functionality),
        ("Context Manager", test_context_manager),
        ("Error Conditions", test_error_conditions),
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
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TopLevelSimulator Python bindings are working correctly.")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())