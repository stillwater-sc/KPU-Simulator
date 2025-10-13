"""
Example demonstrating the new address-based DMA API in Python.

This example shows how to use the unified address space and address-based
DMA operations in the KPU simulator Python bindings.
"""

import stillwater_kpu as kpu
import numpy as np

def main():
    print("=" * 60)
    print("Address-Based DMA API Example")
    print("=" * 60)

    # Configure simulator with all memory hierarchy levels
    config = kpu.SimulatorConfig()

    # Host memory (external to KPU - models system memory)
    config.host_memory_region_count = 1
    config.host_memory_region_capacity_mb = 1024  # 1GB
    config.host_memory_bandwidth_gbps = 50

    # External memory (local to KPU - HBM/GDDR)
    config.memory_bank_count = 2
    config.memory_bank_capacity_mb = 256  # 256MB per bank
    config.memory_bandwidth_gbps = 100

    # On-chip memory hierarchy
    config.l3_tile_count = 2
    config.l3_tile_capacity_kb = 512  # 512KB per tile
    config.l2_bank_count = 4
    config.l2_bank_capacity_kb = 128  # 128KB per bank
    config.scratchpad_count = 2
    config.scratchpad_capacity_kb = 64  # 64KB per scratchpad

    # Compute and data movement engines
    config.compute_tile_count = 2
    config.dma_engine_count = 4
    config.block_mover_count = 2
    config.streamer_count = 4

    # Create simulator
    sim = kpu.KPUSimulator(config)

    print(f"\nMemory Configuration:")
    print(f"  Host memory regions: {sim.get_host_memory_region_count()}")
    print(f"  External banks: {sim.get_memory_bank_count()}")
    print(f"  L3 tiles: {sim.get_l3_tile_count()}")
    print(f"  L2 banks: {sim.get_l2_bank_count()}")
    print(f"  Scratchpads: {sim.get_scratchpad_count()}")
    print(f"  DMA engines: {sim.get_dma_engine_count()}")

    # Print unified address space layout
    print(f"\nUnified Address Space Layout:")
    print(f"  Host[0] base:       0x{sim.get_host_memory_region_base(0):08x}")
    print(f"  External[0] base:   0x{sim.get_external_bank_base(0):08x}")
    print(f"  External[1] base:   0x{sim.get_external_bank_base(1):08x}")
    if sim.get_l3_tile_count() > 0:
        print(f"  L3[0] base:         0x{sim.get_l3_tile_base(0):08x}")
    if sim.get_l2_bank_count() > 0:
        print(f"  L2[0] base:         0x{sim.get_l2_bank_base(0):08x}")
    print(f"  Scratchpad[0] base: 0x{sim.get_scratchpad_base(0):08x}")

    # === Example 1: Host → External → Scratchpad ===
    print("\n" + "=" * 60)
    print("Example 1: Host → External → Scratchpad data flow")
    print("=" * 60)

    # Create test data
    test_data = [float(i) for i in range(256)]

    # Step 1: Write data to host memory
    print("\n1. Writing test data to host memory...")
    sim.write_host_memory(0, 0, test_data)

    # Step 2: DMA from host to external memory
    print("2. DMA transfer: Host → External...")
    host_addr = sim.get_host_memory_region_base(0) + 0
    ext_addr = sim.get_external_bank_base(0) + 0
    transfer_size = len(test_data) * 4  # 4 bytes per float

    transfer_complete = False
    def on_complete():
        nonlocal transfer_complete
        transfer_complete = True
        print("   Transfer complete!")

    sim.dma_host_to_external(0, host_addr, ext_addr, transfer_size, on_complete)
    sim.run_until_idle()

    # Step 3: DMA from external to scratchpad
    print("3. DMA transfer: External → Scratchpad...")
    ext_addr = sim.get_external_bank_base(0) + 0
    scratch_addr = sim.get_scratchpad_base(0) + 0

    transfer_complete = False
    sim.dma_external_to_scratchpad(0, ext_addr, scratch_addr, transfer_size, on_complete)
    sim.run_until_idle()

    # Step 4: Verify data in scratchpad
    print("4. Verifying data in scratchpad...")
    result = sim.read_scratchpad(0, 0, len(test_data))

    if result == test_data:
        print("   ✓ Data verification PASSED")
    else:
        print("   ✗ Data verification FAILED")

    # === Example 2: Multi-level cache hierarchy ===
    print("\n" + "=" * 60)
    print("Example 2: Multi-level cache hierarchy (External → L3 → L2)")
    print("=" * 60)

    if sim.get_l3_tile_count() > 0 and sim.get_l2_bank_count() > 0:
        # Write data to external memory
        cache_data = [float(i * 2) for i in range(128)]
        sim.write_memory_bank(0, 1024, cache_data)

        # Transfer through cache hierarchy
        print("\n1. DMA transfer: External → L3...")
        ext_addr = sim.get_external_bank_base(0) + 1024
        l3_addr = sim.get_l3_tile_base(0) + 0
        transfer_size = len(cache_data) * 4

        sim.dma_external_to_l3(0, ext_addr, l3_addr, transfer_size)
        sim.run_until_idle()

        print("2. Reading back from L3...")
        l3_result = sim.read_l3_tile(0, 0, len(cache_data))

        if l3_result == cache_data:
            print("   ✓ L3 data verification PASSED")
        else:
            print("   ✗ L3 data verification FAILED")
    else:
        print("\nL3/L2 caches not configured in this simulation")

    # === Example 3: Using the primary address-based API ===
    print("\n" + "=" * 60)
    print("Example 3: Primary address-based API (any-to-any transfer)")
    print("=" * 60)

    # The primary API automatically routes based on addresses
    print("\n1. Arbitrary transfer using start_dma_transfer()...")
    src_addr = sim.get_external_bank_base(0) + 0
    dst_addr = sim.get_scratchpad_base(1) + 0  # Different scratchpad
    transfer_size = 128 * 4  # 128 floats

    sim.start_dma_transfer(0, src_addr, dst_addr, transfer_size)
    sim.run_until_idle()
    print("   ✓ Transfer complete (address decoder automatically routed)")

    # === Statistics ===
    print("\n" + "=" * 60)
    print("Simulation Statistics")
    print("=" * 60)
    print(f"Total cycles: {sim.get_current_cycle()}")
    print(f"Elapsed time: {sim.get_elapsed_time_ms():.2f} ms")

    sim.print_component_status()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
