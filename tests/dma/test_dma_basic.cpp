#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

// Test fixture for DMA tests
class DMATestFixture {
public:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> sim;

    DMATestFixture() {
        // Standard test configuration
        config.memory_bank_count = 2;
        config.memory_bank_capacity_mb = 64;
        config.memory_bandwidth_gbps = 8;
        config.scratchpad_count = 2;
        config.scratchpad_capacity_kb = 256;
        config.compute_tile_count = 1;
        config.dma_engine_count = 4; // Multiple DMA engines for various transfer types

        sim = std::make_unique<KPUSimulator>(config);
    }

    // Helper to generate test data
    std::vector<uint8_t> generate_test_pattern(size_t size, uint8_t start_value = 0) {
        std::vector<uint8_t> data(size);
        std::iota(data.begin(), data.end(), start_value);
        return data;
    }

    // Helper to verify data integrity
    bool verify_data(const std::vector<uint8_t>& expected,
                     Address addr, size_t size, size_t memory_bank_id) {
        std::vector<uint8_t> actual(size);
        sim->read_memory_bank(memory_bank_id, addr, actual.data(), size);
        return std::equal(expected.begin(), expected.end(), actual.begin());
    }

    bool verify_scratchpad_data(const std::vector<uint8_t>& expected,
                                Address addr, size_t size, size_t scratchpad_id) {
        std::vector<uint8_t> actual(size);
        sim->read_scratchpad(scratchpad_id, addr, actual.data(), size);
        return std::equal(expected.begin(), expected.end(), actual.begin());
    }
};

TEST_CASE_METHOD(DMATestFixture, "DMA Basic Transfer - External to Scratchpad", "[dma][basic]") {
    const size_t transfer_size = 1024;
    const Address src_addr = 0x1000;
    const Address dst_addr = 0x0;

    // Generate and write test data to external memory
    auto test_data = generate_test_pattern(transfer_size, 0xAA);
    sim->write_memory_bank(0, src_addr, test_data.data(), transfer_size);

    // Start DMA transfer (External[0] -> Scratchpad[0])
    bool transfer_complete = false;
    sim->start_dma_external_to_scratchpad(0, 0, src_addr, 0, dst_addr, transfer_size,
        [&transfer_complete]() { transfer_complete = true; });

    // Process until transfer completes
    while (!transfer_complete) {
        sim->step();
    }

    // Verify data integrity
    REQUIRE(verify_scratchpad_data(test_data, dst_addr, transfer_size, 0));
    REQUIRE_FALSE(sim->is_dma_busy(0));
}

TEST_CASE_METHOD(DMATestFixture, "DMA Basic Transfer - Scratchpad to External", "[dma][basic]") {
    const size_t transfer_size = 2048;
    const Address src_addr = 0x0;
    const Address dst_addr = 0x2000;

    // Generate and write test data to scratchpad
    auto test_data = generate_test_pattern(transfer_size, 0x55);
    sim->write_scratchpad(0, src_addr, test_data.data(), transfer_size);

    // Start DMA transfer (Scratchpad[0] -> External[0])
    bool transfer_complete = false;
    sim->start_dma_scratchpad_to_external(0, 0, src_addr, 0, dst_addr, transfer_size,
        [&transfer_complete]() { transfer_complete = true; });

    // Process until transfer completes
    while (!transfer_complete) {
        sim->step();
    }

    // Verify data integrity
    REQUIRE(verify_data(test_data, dst_addr, transfer_size, 0));
    REQUIRE_FALSE(sim->is_dma_busy(1));
}

TEST_CASE_METHOD(DMATestFixture, "DMA Queue Management - Multiple Transfers", "[dma][queue]") {
    const size_t transfer_size = 512;

    // Prepare multiple test datasets
    auto data1 = generate_test_pattern(transfer_size, 0x11);
    auto data2 = generate_test_pattern(transfer_size, 0x22);
    auto data3 = generate_test_pattern(transfer_size, 0x33);

    // Write test data to external memory
    sim->write_memory_bank(0, 0x1000, data1.data(), transfer_size);
    sim->write_memory_bank(0, 0x1000 + transfer_size, data2.data(), transfer_size);
    sim->write_memory_bank(0, 0x1000 + 2*transfer_size, data3.data(), transfer_size);

    // Queue multiple transfers
    int completions = 0;
    auto completion_callback = [&completions]() { completions++; };

    sim->start_dma_external_to_scratchpad(0, 0, 0x1000, 0, 0x0, transfer_size, completion_callback);
    sim->start_dma_external_to_scratchpad(0, 0, 0x1000 + transfer_size, 0, transfer_size, transfer_size, completion_callback);
    sim->start_dma_external_to_scratchpad(0, 0, 0x1000 + 2*transfer_size, 0, 2*transfer_size, transfer_size, completion_callback);

    REQUIRE(sim->is_dma_busy(0));

    // Process until all transfers complete
    while (completions < 3) {
        sim->step();
    }

    // Verify all data transferred correctly (FIFO order)
    REQUIRE(verify_scratchpad_data(data1, 0x0, transfer_size, 0));
    REQUIRE(verify_scratchpad_data(data2, transfer_size, transfer_size, 0));
    REQUIRE(verify_scratchpad_data(data3, 2*transfer_size, transfer_size, 0));
    REQUIRE_FALSE(sim->is_dma_busy(0));
}

TEST_CASE_METHOD(DMATestFixture, "DMA Data Integrity - Various Sizes", "[dma][integrity]") {
    std::vector<size_t> test_sizes = {1, 16, 64, 256, 1024, 4096, 65536};

    for (size_t size : test_sizes) {
        SECTION("Transfer size: " + std::to_string(size) + " bytes") {
            if (size > sim->get_scratchpad_capacity(0)) {
                SKIP("Transfer size exceeds scratchpad capacity");
            }

            auto test_data = generate_test_pattern(size, static_cast<uint8_t>(size & 0xFF));
            sim->write_memory_bank(0, 0, test_data.data(), size);

            bool complete = false;
            sim->start_dma_external_to_scratchpad(0, 0, 0, 0, 0, size, [&complete]() { complete = true; });

            while (!complete) {
                sim->step();
            }

            REQUIRE(verify_scratchpad_data(test_data, 0, size, 0));
        }
    }
}

TEST_CASE_METHOD(DMATestFixture, "DMA Error Handling - Invalid Addresses", "[dma][error]") {
    const size_t transfer_size = 1024;

    SECTION("Source address out of bounds") {
        Address invalid_src = sim->get_memory_bank_capacity(0) + 1000;

        // Queuing should succeed (lazy validation)
        sim->start_dma_external_to_scratchpad(0, 0, invalid_src, 0, 0, transfer_size);

        // The error should occur during processing
        REQUIRE_THROWS_AS(
            sim->step(),
            std::out_of_range
        );
    }

    SECTION("Destination address out of bounds") {
        Address invalid_dst = sim->get_scratchpad_capacity(0) + 1000;

        sim->start_dma_external_to_scratchpad(0, 0, 0, 0, invalid_dst, transfer_size);

        // The error should occur during processing
        REQUIRE_THROWS_AS(
            sim->step(),
            std::out_of_range
        );
    }

    SECTION("Transfer size exceeds destination capacity") {
        size_t oversized_transfer = sim->get_scratchpad_capacity(0) + 1024;

        sim->start_dma_external_to_scratchpad(0, 0, 0, 0, 0, oversized_transfer);

        REQUIRE_THROWS_AS(
            sim->step(),
            std::out_of_range
        );
    }
}

TEST_CASE_METHOD(DMATestFixture, "DMA Error Handling - Invalid IDs", "[dma][error]") {
    SECTION("Invalid DMA engine ID") {
        REQUIRE_THROWS_AS(
            sim->start_dma_external_to_scratchpad(99, 0, 0, 0, 0, 1024),
            std::out_of_range
        );

        REQUIRE_THROWS_AS(
            sim->is_dma_busy(99),
            std::out_of_range
        );
    }
}

TEST_CASE_METHOD(DMATestFixture, "DMA Reset Functionality", "[dma][reset]") {
    const size_t transfer_size = 1024;
    auto test_data = generate_test_pattern(transfer_size);

    // Queue a transfer
    sim->write_memory_bank(0, 0, test_data.data(), transfer_size);
    sim->start_dma_external_to_scratchpad(0, 0, 0, 0, 0, transfer_size);

    REQUIRE(sim->is_dma_busy(0));

    // Reset the simulator
    sim->reset();

    // DMA should no longer be busy
    REQUIRE_FALSE(sim->is_dma_busy(0));
}

TEST_CASE_METHOD(DMATestFixture, "DMA Concurrent Operations", "[dma][concurrent]") {
    const size_t transfer_size = 1024;

    // Prepare test data for different DMA engines
    auto data1 = generate_test_pattern(transfer_size, 0xAA);
    auto data2 = generate_test_pattern(transfer_size, 0xBB);

    // Write to different memory banks
    sim->write_memory_bank(0, 0, data1.data(), transfer_size);
    sim->write_memory_bank(1, 0, data2.data(), transfer_size);

    // Start concurrent transfers using different DMA engines
    bool transfer1_complete = false;
    bool transfer2_complete = false;

    // DMA 0: Bank0 -> Scratchpad0
    sim->start_dma_external_to_scratchpad(0, 0, 0, 0, 0, transfer_size,
        [&transfer1_complete]() { transfer1_complete = true; });

    // DMA 1: Bank1 -> Scratchpad0 (different section)
    if (sim->get_scratchpad_capacity(0) >= 2 * transfer_size) {
        sim->start_dma_external_to_scratchpad(1, 1, 0, 0, transfer_size, transfer_size,
            [&transfer2_complete]() { transfer2_complete = true; });
    }

    REQUIRE(sim->is_dma_busy(0));
    if (sim->get_scratchpad_capacity(0) >= 2 * transfer_size) {
        REQUIRE(sim->is_dma_busy(1));
    }

    // Process until both complete
    while (!transfer1_complete || (!transfer2_complete && sim->get_scratchpad_capacity(0) >= 2 * transfer_size)) {
        sim->step();
    }

    // Verify both transfers completed correctly
    REQUIRE(verify_scratchpad_data(data1, 0, transfer_size, 0));
    if (sim->get_scratchpad_capacity(0) >= 2 * transfer_size) {
        REQUIRE(verify_scratchpad_data(data2, transfer_size, transfer_size, 0));
    }
}

TEST_CASE_METHOD(DMATestFixture, "DMA Matrix Data Movement", "[dma][matrix]") {
    // Simulate moving matrix data for tensor operations
    const size_t matrix_rows = 32;
    const size_t matrix_cols = 32;
    const size_t matrix_size = matrix_rows * matrix_cols * sizeof(float);

    if (matrix_size > sim->get_scratchpad_capacity(0)) {
        SKIP("Matrix too large for scratchpad");
    }

    // Generate matrix data
    std::vector<float> matrix_a(matrix_rows * matrix_cols);
    std::vector<float> matrix_b(matrix_rows * matrix_cols);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::generate(matrix_a.begin(), matrix_a.end(), [&]() { return dis(gen); });
    std::generate(matrix_b.begin(), matrix_b.end(), [&]() { return dis(gen); });

    // Write matrices to external memory
    sim->write_memory_bank(0, 0, matrix_a.data(), matrix_size);
    sim->write_memory_bank(0, matrix_size, matrix_b.data(), matrix_size);

    // Transfer matrices to scratchpad
    bool transfer_a_complete = false;
    bool transfer_b_complete = false;

    sim->start_dma_external_to_scratchpad(0, 0, 0, 0, 0, matrix_size,
        [&transfer_a_complete]() { transfer_a_complete = true; });
    sim->start_dma_external_to_scratchpad(0, 0, matrix_size, 0, matrix_size, matrix_size,
        [&transfer_b_complete]() { transfer_b_complete = true; });

    // Process transfers
    while (!transfer_a_complete || !transfer_b_complete) {
        sim->step();
    }

    // Verify matrix data integrity
    std::vector<float> read_matrix_a(matrix_rows * matrix_cols);
    std::vector<float> read_matrix_b(matrix_rows * matrix_cols);

    sim->read_scratchpad(0, 0, read_matrix_a.data(), matrix_size);
    sim->read_scratchpad(0, matrix_size, read_matrix_b.data(), matrix_size);

    REQUIRE(std::equal(matrix_a.begin(), matrix_a.end(), read_matrix_a.begin()));
    REQUIRE(std::equal(matrix_b.begin(), matrix_b.end(), read_matrix_b.begin()));
}