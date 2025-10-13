#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

TEST_CASE("DMA Debug - Configuration Check", "[dma][debug]") {
    KPUSimulator::Config config;
    config.memory_bank_count = 4;
    config.memory_bank_capacity_mb = 128;
    config.memory_bandwidth_gbps = 32;
    config.scratchpad_count = 4;
    config.scratchpad_capacity_kb = 1024;
    config.compute_tile_count = 2;
    config.dma_engine_count = 8;

    KPUSimulator sim(config);

    std::cout << "\n=== DMA Configuration Debug ===\n";
    std::cout << "Memory banks: " << sim.get_memory_bank_count() << "\n";
    std::cout << "Scratchpads: " << sim.get_scratchpad_count() << "\n";
    std::cout << "DMA engines: " << sim.get_dma_engine_count() << "\n";
    std::cout << "Memory bank capacity: " << sim.get_memory_bank_capacity(0) << " bytes\n";
    std::cout << "Scratchpad capacity: " << sim.get_scratchpad_capacity(0) << " bytes\n";

    // Print component status before any operations
    sim.print_component_status();

    REQUIRE(sim.get_memory_bank_count() == 4);
    REQUIRE(sim.get_scratchpad_count() == 4);
    REQUIRE(sim.get_dma_engine_count() == 8);
}

TEST_CASE("DMA Debug - Simple Transfer", "[dma][debug]") {
    KPUSimulator::Config config;
    config.memory_bank_count = 2;
    config.memory_bank_capacity_mb = 64;
    config.memory_bandwidth_gbps = 8;
    config.scratchpad_count = 2;
    config.scratchpad_capacity_kb = 256;
    config.compute_tile_count = 1;
    config.dma_engine_count = 4;

    KPUSimulator sim(config);

    const size_t transfer_size = 1024; // 1KB
    std::vector<uint8_t> test_data(transfer_size);
    std::iota(test_data.begin(), test_data.end(), 0);

    std::cout << "\n=== Simple DMA Transfer Debug ===\n";

    // Write test data to memory bank 0
    sim.write_memory_bank(0, 0, test_data.data(), transfer_size);
    std::cout << "Written " << transfer_size << " bytes to memory bank 0\n";

    // Check DMA 0 configuration (should be External->Scratchpad)
    std::cout << "DMA 0 busy before transfer: " << (sim.is_dma_busy(0) ? "YES" : "NO") << "\n";

    // Start transfer using DMA 0
    Address global_src = sim.get_external_bank_base(0);
    Address global_dst = sim.get_scratchpad_base(0);

    bool transfer_complete = false;
    sim.dma_external_to_scratchpad(0, global_src, global_dst, transfer_size,
        [&transfer_complete]() {
            std::cout << "Transfer completion callback called\n";
            transfer_complete = true;
        });

    std::cout << "DMA 0 busy after queuing: " << (sim.is_dma_busy(0) ? "YES" : "NO") << "\n";

    // Process transfers step by step with debugging
    size_t steps = 0;
    const size_t max_steps = 100; // Prevent infinite loops

    while (!transfer_complete && steps < max_steps) {
        sim.step();
        steps++;

        if (steps <= 10 || steps % 10 == 0) {
            std::cout << "Step " << steps << ": DMA busy = "
                      << (sim.is_dma_busy(0) ? "YES" : "NO")
                      << ", Complete = " << (transfer_complete ? "YES" : "NO") << "\n";
        }
    }

    std::cout << "Total steps: " << steps << "\n";
    std::cout << "Transfer completed: " << (transfer_complete ? "YES" : "NO") << "\n";

    if (transfer_complete) {
        // Verify data integrity
        std::vector<uint8_t> read_data(transfer_size);
        sim.read_scratchpad(0, 0, read_data.data(), transfer_size);

        bool data_correct = std::equal(test_data.begin(), test_data.end(), read_data.begin());
        std::cout << "Data integrity: " << (data_correct ? "PASS" : "FAIL") << "\n";

        REQUIRE(data_correct);
    } else {
        std::cout << "ERROR: Transfer did not complete within " << max_steps << " steps\n";
        FAIL("Transfer did not complete");
    }

    REQUIRE(transfer_complete);
}