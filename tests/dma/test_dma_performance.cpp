#include <iostream>
#include <numeric>
#include <chrono>
#include <algorithm>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

class DMAPerformanceFixture {
public:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> sim;

    DMAPerformanceFixture() {
        // Configuration optimized for performance testing
        config.memory_bank_count = 4;
        config.memory_bank_capacity_mb = 128;
        config.memory_bandwidth_gbps = 32;
        config.scratchpad_count = 4;
        config.scratchpad_capacity_kb = 1024; // Larger scratchpads
        config.compute_tile_count = 2;
        config.dma_engine_count = 8; // More DMA engines for parallelism

        sim = std::make_unique<KPUSimulator>(config);
    }

    // Helper to measure transfer throughput
    double measure_transfer_throughput(size_t transfer_size, size_t dma_id = 0) {
        // Prepare test data
        std::vector<uint8_t> test_data(transfer_size);
        std::iota(test_data.begin(), test_data.end(), 0);

        sim->write_memory_bank(0, 0, test_data.data(), transfer_size);

        auto start_time = std::chrono::high_resolution_clock::now();

        bool complete = false;
        sim->start_dma_external_to_scratchpad(dma_id, 0, 0, 0, 0, transfer_size,
            [&complete]() { complete = true; });

        size_t cycles = 0;
        const size_t max_cycles = 10000; // Prevent infinite loops
        while (!complete && cycles < max_cycles) {
            sim->step();
            cycles++;
        }

        if (!complete) {
            throw std::runtime_error("DMA transfer did not complete within reasonable time");
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // Return MB/s (based on wall-clock time)
        double seconds = duration.count() / 1000000.0;
        return (transfer_size / (1024.0 * 1024.0)) / seconds;
    }
};

TEST_CASE_METHOD(DMAPerformanceFixture, "DMA Performance - Transfer Size Scaling", "[dma][performance]") {
    std::vector<size_t> transfer_sizes = {
        1024,       // 1KB
        16384,      // 16KB
        65536,      // 64KB
        262144,     // 256KB
        1048576     // 1MB
    };

    std::cout << "\nDMA Transfer Performance Analysis:\n";
    std::cout << "Size\t\tThroughput (MB/s)\tCycles\n";

    for (size_t size : transfer_sizes) {
        if (size > sim->get_scratchpad_capacity(0)) {
            continue;
        }

        // Measure performance
        std::vector<uint8_t> test_data(size);
        std::iota(test_data.begin(), test_data.end(), 0);

        sim->write_memory_bank(0, 0, test_data.data(), size);

        auto start_time = std::chrono::high_resolution_clock::now();
        size_t start_cycle = sim->get_current_cycle();

        bool complete = false;
        sim->start_dma_external_to_scratchpad(0, 0, 0, 0, 0, size,
            [&complete]() { complete = true; });

        while (!complete) {
            sim->step();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        size_t end_cycle = sim->get_current_cycle();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double seconds = duration.count() / 1000000.0;
        double throughput = (size / (1024.0 * 1024.0)) / seconds;

        std::cout << size << " bytes\t" << throughput << " MB/s\t\t" << (end_cycle - start_cycle) << " cycles\n";

        // Basic performance validation
        REQUIRE(throughput > 0);
        REQUIRE((end_cycle - start_cycle) > 0);

        sim->reset();
    }
}

TEST_CASE_METHOD(DMAPerformanceFixture, "DMA Performance - Concurrent Transfer Scaling", "[dma][performance]") {
    const size_t transfer_size = 65536; // 64KB per transfer
    const size_t max_concurrent = std::min(static_cast<size_t>(4), config.dma_engine_count);

    std::cout << "\nConcurrent DMA Transfer Analysis:\n";
    std::cout << "Concurrent DMAs\tTotal Throughput (MB/s)\tEfficiency\n";

    for (size_t concurrent_count = 1; concurrent_count <= max_concurrent; ++concurrent_count) {
        if (transfer_size * concurrent_count > sim->get_scratchpad_capacity(0)) {
            continue;
        }

        // Prepare test data for each transfer
        std::vector<std::vector<uint8_t>> test_data_sets(concurrent_count);
        for (size_t i = 0; i < concurrent_count; ++i) {
            test_data_sets[i].resize(transfer_size);
            std::iota(test_data_sets[i].begin(), test_data_sets[i].end(), static_cast<uint8_t>(i));
            sim->write_memory_bank(i % config.memory_bank_count, i * transfer_size,
                                 test_data_sets[i].data(), transfer_size);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Start concurrent transfers
        std::vector<bool> completions(concurrent_count, false);
        for (size_t i = 0; i < concurrent_count; ++i) {
            size_t src_bank = i % config.memory_bank_count;
            size_t dst_offset = i * transfer_size;

            sim->start_dma_external_to_scratchpad(i, src_bank, i * transfer_size, 0, dst_offset, transfer_size,
                [&completions, i]() { completions[i] = true; });
        }

        // Wait for all transfers to complete
        while (!std::all_of(completions.begin(), completions.end(), [](bool c) { return c; })) {
            sim->step();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        double seconds = duration.count() / 1000000.0;
        double total_mb = (transfer_size * concurrent_count) / (1024.0 * 1024.0);
        double total_throughput = total_mb / seconds;
        double single_throughput = (transfer_size / (1024.0 * 1024.0)) / seconds; // Per-transfer throughput
        double efficiency = total_throughput / (single_throughput * concurrent_count);

        std::cout << concurrent_count << "\t\t" << total_throughput << " MB/s\t\t" << efficiency << "\n";

        // Validate that concurrent transfers completed correctly
        for (size_t i = 0; i < concurrent_count; ++i) {
            std::vector<uint8_t> read_data(transfer_size);
            sim->read_scratchpad(0, i * transfer_size, read_data.data(), transfer_size);
            REQUIRE(std::equal(test_data_sets[i].begin(), test_data_sets[i].end(), read_data.begin()));
        }

        sim->reset();
    }
}

TEST_CASE_METHOD(DMAPerformanceFixture, "DMA Performance - Memory Bank Distribution", "[dma][performance]") {
    const size_t transfer_size = 262144; // 256KB
    const size_t num_banks = config.memory_bank_count;

    std::cout << "\nMemory Bank Distribution Performance:\n";
    std::cout << "Strategy\t\tThroughput (MB/s)\n";

    // Test 1: Single bank source
    SECTION("Single Bank Source") {
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<uint8_t> test_data(transfer_size);
        std::iota(test_data.begin(), test_data.end(), 0);
        sim->write_memory_bank(0, 0, test_data.data(), transfer_size);

        bool complete = false;
        sim->start_dma_external_to_scratchpad(0, 0, 0, 0, 0, transfer_size,
            [&complete]() { complete = true; });

        while (!complete) {
            sim->step();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double throughput = (transfer_size / (1024.0 * 1024.0)) / (duration.count() / 1000000.0);

        std::cout << "Single Bank\t\t" << throughput << " MB/s\n";
        REQUIRE(throughput > 0);
    }

    // Test 2: Distributed across banks
    SECTION("Distributed Banks") {
        if (num_banks < 2) {
            SKIP("Need at least 2 memory banks for distribution test");
        }

        const size_t per_bank_size = transfer_size / num_banks;
        if (per_bank_size * num_banks > sim->get_scratchpad_capacity(0)) {
            SKIP("Total transfer size exceeds scratchpad capacity");
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Prepare data across multiple banks
        std::vector<bool> completions(num_banks, false);
        for (size_t bank = 0; bank < num_banks; ++bank) {
            std::vector<uint8_t> test_data(per_bank_size);
            std::iota(test_data.begin(), test_data.end(), static_cast<uint8_t>(bank));
            sim->write_memory_bank(bank, 0, test_data.data(), per_bank_size);

            // Use different DMA engines for different banks
            size_t dma_id = bank % config.dma_engine_count;
            sim->start_dma_external_to_scratchpad(dma_id, bank, 0, 0, bank * per_bank_size, per_bank_size,
                [&completions, bank]() { completions[bank] = true; });
        }

        while (!std::all_of(completions.begin(), completions.end(), [](bool c) { return c; })) {
            sim->step();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double total_size_mb = (per_bank_size * num_banks) / (1024.0 * 1024.0);
        double throughput = total_size_mb / (duration.count() / 1000000.0);

        std::cout << "Distributed\t\t" << throughput << " MB/s\n";
        REQUIRE(throughput > 0);
    }
}

TEST_CASE_METHOD(DMAPerformanceFixture, "DMA Performance - Large Dataset Streaming", "[dma][performance][streaming]") {
    // Simulate streaming large datasets in chunks (L3->L2 simulation)
    const size_t total_dataset_size = 8 * 1024 * 1024; // 8MB total dataset
    const size_t chunk_size = 256 * 1024; // 256KB chunks
    const size_t num_chunks = total_dataset_size / chunk_size;

    if (chunk_size > sim->get_scratchpad_capacity(0)) {
        SKIP("Chunk size exceeds scratchpad capacity");
    }

    std::cout << "\nLarge Dataset Streaming Performance:\n";
    std::cout << "Streaming " << (total_dataset_size / (1024*1024)) << "MB in "
              << (chunk_size / 1024) << "KB chunks\n";

    // Prepare dataset in external memory
    std::vector<uint8_t> full_dataset(total_dataset_size);
    std::iota(full_dataset.begin(), full_dataset.end(), 0);

    // Write dataset across memory banks
    size_t bytes_per_bank = total_dataset_size / config.memory_bank_count;
    for (size_t bank = 0; bank < config.memory_bank_count; ++bank) {
        size_t start_idx = bank * bytes_per_bank;
        size_t size = (bank == config.memory_bank_count - 1) ?
                      (total_dataset_size - start_idx) : bytes_per_bank;
        sim->write_memory_bank(bank, 0, full_dataset.data() + start_idx, size);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Stream data in chunks
    size_t chunks_completed = 0;
    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        size_t src_bank = chunk % config.memory_bank_count;
        size_t src_offset = (chunk / config.memory_bank_count) * chunk_size;
        size_t dma_id = chunk % config.dma_engine_count;

        bool chunk_complete = false;
        sim->start_dma_external_to_scratchpad(dma_id, src_bank, src_offset, 0, 0, chunk_size,
            [&chunk_complete, &chunks_completed]() {
                chunk_complete = true;
                chunks_completed++;
            });

        // Process this chunk
        while (!chunk_complete) {
            sim->step();
        }

        // Verify chunk data integrity
        std::vector<uint8_t> chunk_data(chunk_size);
        sim->read_scratchpad(0, 0, chunk_data.data(), chunk_size);

        size_t dataset_offset = chunk * chunk_size;
        bool data_correct = std::equal(
            chunk_data.begin(), chunk_data.end(),
            full_dataset.begin() + dataset_offset
        );
        REQUIRE(data_correct);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    double total_mb = total_dataset_size / (1024.0 * 1024.0);
    double throughput = total_mb / (duration.count() / 1000000.0);
    double avg_chunk_time = (duration.count() / 1000.0) / num_chunks; // ms per chunk

    std::cout << "Total Throughput: " << throughput << " MB/s\n";
    std::cout << "Average Chunk Time: " << avg_chunk_time << " ms\n";
    std::cout << "Chunks Completed: " << chunks_completed << "/" << num_chunks << "\n";

    REQUIRE(chunks_completed == num_chunks);
    REQUIRE(throughput > 0);
}

TEST_CASE_METHOD(DMAPerformanceFixture, "DMA Performance - Benchmark Test", "[dma][performance][benchmark]") {
    const size_t transfer_size = 1024 * 1024; // 1MB
    const size_t num_iterations = 10;

    std::vector<uint8_t> test_data(transfer_size);
    std::iota(test_data.begin(), test_data.end(), 0);

    std::vector<double> throughputs;
    std::vector<size_t> cycle_counts;

    std::cout << "\nDMA Benchmark Results (1MB transfers):\n";
    std::cout << "Iteration\tCycles\t\tThroughput (MB/s)\n";

    for (size_t iter = 0; iter < num_iterations; ++iter) {
        sim->write_memory_bank(0, 0, test_data.data(), transfer_size);

        auto start_time = std::chrono::high_resolution_clock::now();
        size_t start_cycle = sim->get_current_cycle();

        bool complete = false;
        sim->start_dma_external_to_scratchpad(0, 0, 0, 0, 0, transfer_size,
            [&complete]() { complete = true; });

        while (!complete) {
            sim->step();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        size_t end_cycle = sim->get_current_cycle();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double seconds = duration.count() / 1000000.0;
        double throughput = (transfer_size / (1024.0 * 1024.0)) / seconds;
        size_t cycles = end_cycle - start_cycle;

        throughputs.push_back(throughput);
        cycle_counts.push_back(cycles);

        std::cout << iter + 1 << "\t\t" << cycles << "\t\t" << throughput << "\n";

        sim->reset();
    }

    // Calculate statistics
    double avg_throughput = std::accumulate(throughputs.begin(), throughputs.end(), 0.0) / num_iterations;
    double avg_cycles = std::accumulate(cycle_counts.begin(), cycle_counts.end(), 0.0) / num_iterations;

    std::cout << "\nAverage Throughput: " << avg_throughput << " MB/s\n";
    std::cout << "Average Cycles: " << avg_cycles << "\n";

    // Basic performance validation
    REQUIRE(avg_throughput > 0);
    REQUIRE(avg_cycles > 0);
    REQUIRE(throughputs.size() == num_iterations);
};