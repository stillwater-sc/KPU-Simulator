#!/usr/bin/env python
"""
Quick test script for the address-based DMA API.
Run this to verify your Python bindings are working.
"""

import sys
import os

# Auto-detect build directory and add to path
possible_paths = [
    './build/src/bindings/python/Release',
    './build_msvc/src/bindings/python/Release',
    './build_msvc/src/bindings/python/Debug',
    '../build/src/bindings/python/Release',
    '../build_msvc/src/bindings/python/Release',
]

build_path = None
for path in possible_paths:
    if os.path.exists(path):
        build_path = os.path.abspath(path)
        sys.path.insert(0, build_path)
        print(f"Using Python bindings from: {build_path}")
        break

if not build_path:
    print("ERROR: Could not find Python bindings!")
    print("Expected locations:")
    for path in possible_paths:
        print(f"  {os.path.abspath(path)}")
    print("\nPlease build first with:")
    print("  cmake --build . --target stillwater_kpu")
    sys.exit(1)

try:
    import stillwater_kpu as kpu
    print(f"✓ Successfully imported stillwater_kpu\n")
except ImportError as e:
    print(f"ERROR: Failed to import stillwater_kpu: {e}")
    sys.exit(1)

def test_basic_dma():
    """Test basic DMA transfer External → Scratchpad"""
    print("=" * 70)
    print(" Testing Address-Based DMA API")
    print("=" * 70)

    # Create simulator
    config = kpu.SimulatorConfig()
    config.memory_bank_count = 2
    config.memory_bank_capacity_mb = 64
    config.scratchpad_count = 2
    config.scratchpad_capacity_kb = 256
    config.dma_engine_count = 4

    sim = kpu.KPUSimulator(config)

    print(f"\n[CONFIG] Simulator initialized:")
    print(f"         External banks: {sim.get_memory_bank_count()}")
    print(f"         Scratchpads: {sim.get_scratchpad_count()}")
    print(f"         DMA engines: {sim.get_dma_engine_count()}")

    # Address space
    print(f"\n[MEMORY] Address Space:")
    print(f"         External[0]: 0x{sim.get_external_bank_base(0):08x}")
    print(f"         Scratchpad[0]: 0x{sim.get_scratchpad_base(0):08x}")

    # Prepare test data
    test_data = [float(i) for i in range(256)]
    print(f"\n[DATA]   Generated {len(test_data)} test values")

    # Write to external memory
    print(f"\n[WRITE]  Writing to external memory bank 0...")
    sim.write_memory_bank(0, 0, test_data)
    print(f"         ✓ Written {len(test_data)} floats ({len(test_data) * 4} bytes)")

    # Compute global addresses for DMA
    src_addr = sim.get_external_bank_base(0) + 0
    dst_addr = sim.get_scratchpad_base(0) + 0
    transfer_size = len(test_data) * 4  # 4 bytes per float

    print(f"\n[DMA]    Initiating transfer:")
    print(f"         Source:      0x{src_addr:08x} (External bank 0)")
    print(f"         Destination: 0x{dst_addr:08x} (Scratchpad 0)")
    print(f"         Size:        {transfer_size} bytes ({len(test_data)} floats)")

    # Start DMA transfer with callback
    complete = [False]
    def on_complete():
        complete[0] = True

    sim.dma_external_to_scratchpad(0, src_addr, dst_addr, transfer_size, on_complete)
    print(f"         ✓ DMA transfer queued on engine 0")

    # Run simulation
    print(f"\n[SIM]    Running simulation...")
    sim.run_until_idle()

    if complete[0]:
        print(f"         ✓ DMA transfer completed")
    else:
        print(f"         ✗ DMA transfer did not complete!")
        return False

    # Verify data
    print(f"\n[VERIFY] Reading back from scratchpad...")
    result = sim.read_scratchpad(0, 0, len(test_data))

    matches = 0
    mismatches = 0
    for i in range(min(len(result), len(test_data))):
        if result[i] == test_data[i]:
            matches += 1
        else:
            mismatches += 1

    print(f"         Matches: {matches}/{len(test_data)}")
    if mismatches > 0:
        print(f"         ✗ Mismatches: {mismatches}")
        return False
    else:
        print(f"         ✓ All data verified correctly!")

    # Print statistics
    print(f"\n[STATS]  Simulation Statistics:")
    print(f"         Cycles: {sim.get_current_cycle()}")
    print(f"         Time: {sim.get_elapsed_time_ms():.3f} ms")

    print("\n" + "=" * 70)
    print(" ✓ All tests PASSED!")
    print("=" * 70)
    return True

def test_multi_stage_pipeline():
    """Test multi-stage pipeline: Host → External → Scratchpad"""
    print("\n" + "=" * 70)
    print(" Testing Multi-Stage Pipeline")
    print("=" * 70)

    config = kpu.SimulatorConfig()
    config.host_memory_region_count = 1
    config.memory_bank_count = 2
    config.scratchpad_count = 2
    config.dma_engine_count = 4

    sim = kpu.KPUSimulator(config)

    # Test data
    test_data = [float(i * 2) for i in range(128)]

    print(f"\n[STAGE 1] Host → External")
    sim.write_host_memory(0, 0, test_data)

    host_addr = sim.get_host_memory_region_base(0)
    ext_addr = sim.get_external_bank_base(0)
    size = len(test_data) * 4

    sim.dma_host_to_external(0, host_addr, ext_addr, size)
    sim.run_until_idle()
    print(f"          ✓ Completed")

    print(f"\n[STAGE 2] External → Scratchpad")
    scratch_addr = sim.get_scratchpad_base(0)

    sim.dma_external_to_scratchpad(1, ext_addr, scratch_addr, size)
    sim.run_until_idle()
    print(f"          ✓ Completed")

    print(f"\n[VERIFY]  Final data check...")
    result = sim.read_scratchpad(0, 0, len(test_data))

    if result == test_data:
        print(f"          ✓ Pipeline data verified!")
        print("\n" + "=" * 70)
        print(" ✓ Pipeline test PASSED!")
        print("=" * 70)
        return True
    else:
        print(f"          ✗ Data mismatch!")
        return False

if __name__ == "__main__":
    print("\n🚀 KPU Simulator Python Bindings Test Suite\n")

    success = True

    # Test 1: Basic DMA
    try:
        if not test_basic_dma():
            success = False
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 2: Multi-stage pipeline
    try:
        if not test_multi_stage_pipeline():
            success = False
    except Exception as e:
        print(f"\n✗ Pipeline test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "=" * 70)
    if success:
        print(" 🎉 ALL TESTS PASSED!")
    else:
        print(" ❌ SOME TESTS FAILED")
    print("=" * 70 + "\n")

    sys.exit(0 if success else 1)
