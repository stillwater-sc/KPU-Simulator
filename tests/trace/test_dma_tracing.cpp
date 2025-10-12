#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <sw/kpu/components/dma_engine.hpp>
#include <sw/memory/external_memory.hpp>
#include <sw/kpu/components/scratchpad.hpp>
#include <sw/kpu/components/l3_tile.hpp>
#include <sw/kpu/components/l2_bank.hpp>
#include <sw/trace/trace_logger.hpp>
#include <sw/trace/trace_exporter.hpp>

using namespace sw;
using namespace sw::kpu;
using namespace sw::trace;

// Test fixture for DMA tracing tests
class DMATracingFixture {
public:
    std::vector<ExternalMemory> memory_banks;
    std::vector<L3Tile> l3_tiles;  // Empty for these tests
    std::vector<L2Bank> l2_banks;  // Empty for these tests
    std::vector<Scratchpad> scratchpads;
    DMAEngine dma_engine;
    TraceLogger& logger;

    DMATracingFixture()
        : dma_engine(0, 1.0, 100.0)  // Engine 0, 1 GHz, 100 GB/s
        , logger(TraceLogger::instance())
    {
        // Create 2 memory banks of 64MB each
        memory_banks.emplace_back(64, 100);  // 64 MB capacity, 100 Gbps bandwidth
        memory_banks.emplace_back(64, 100);

        // Create 2 scratchpads of 256KB each
        scratchpads.emplace_back(256);  // 256 KB capacity
        scratchpads.emplace_back(256);

        // L3 and L2 remain empty for these tests (only testing EXTERNAL <-> SCRATCHPAD)

        // Reset and configure tracing
        logger.clear();
        logger.set_enabled(true);
        dma_engine.enable_tracing(true, &logger);
    }

    ~DMATracingFixture() {
        // Leave logger enabled for inspection after tests
    }

    // Helper to generate test data
    std::vector<uint8_t> generate_test_pattern(size_t size, uint8_t start_value = 0) {
        std::vector<uint8_t> data(size);
        std::iota(data.begin(), data.end(), start_value);
        return data;
    }
};

TEST_CASE_METHOD(DMATracingFixture, "Trace: Single DMA Transfer - External to Scratchpad", "[trace][dma]") {
    const size_t transfer_size = 4096;
    const Address src_addr = 0x1000;
    const Address dst_addr = 0x0;

    // Generate and write test data
    auto test_data = generate_test_pattern(transfer_size, 0xAA);
    memory_banks[0].write(src_addr, test_data.data(), transfer_size);

    // Set initial cycle
    dma_engine.set_current_cycle(1000);

    size_t initial_trace_count = logger.get_trace_count();

    // Enqueue transfer
    dma_engine.enqueue_transfer(
        DMAEngine::MemoryType::EXTERNAL, 0, src_addr,
        DMAEngine::MemoryType::SCRATCHPAD, 0, dst_addr,
        transfer_size
    );

    // Should have logged the issue
    REQUIRE(logger.get_trace_count() == initial_trace_count + 1);

    // Process the transfer
    dma_engine.process_transfers(memory_banks, l3_tiles, l2_banks, scratchpads);

    // Should have logged the completion
    REQUIRE(logger.get_trace_count() == initial_trace_count + 2);

    // Get traces for this DMA engine
    auto dma_traces = logger.get_component_traces(ComponentType::DMA_ENGINE, 0);
    REQUIRE(dma_traces.size() >= 2);

    // Verify the last two traces (issue and completion)
    auto& issue_trace = dma_traces[dma_traces.size() - 2];
    auto& complete_trace = dma_traces[dma_traces.size() - 1];

    // Verify issue trace
    REQUIRE(issue_trace.component_type == ComponentType::DMA_ENGINE);
    REQUIRE(issue_trace.component_id == 0);
    REQUIRE(issue_trace.transaction_type == TransactionType::TRANSFER);
    REQUIRE(issue_trace.cycle_issue == 1000);
    REQUIRE(issue_trace.status == TransactionStatus::ISSUED);

    // Verify completion trace
    REQUIRE(complete_trace.component_type == ComponentType::DMA_ENGINE);
    REQUIRE(complete_trace.component_id == 0);
    REQUIRE(complete_trace.transaction_type == TransactionType::TRANSFER);
    REQUIRE(complete_trace.status == TransactionStatus::COMPLETED);
    REQUIRE(complete_trace.cycle_complete > complete_trace.cycle_issue);

    // Verify payload data
    REQUIRE(std::holds_alternative<DMAPayload>(complete_trace.payload));
    const auto& payload = std::get<DMAPayload>(complete_trace.payload);
    REQUIRE(payload.bytes_transferred == transfer_size);
    REQUIRE(payload.source.address == src_addr);
    REQUIRE(payload.destination.address == dst_addr);

    std::cout << "\n=== DMA Transfer Trace ===" << std::endl;
    std::cout << "Transaction ID: " << complete_trace.transaction_id << std::endl;
    std::cout << "Issue Cycle: " << complete_trace.cycle_issue << std::endl;
    std::cout << "Complete Cycle: " << complete_trace.cycle_complete << std::endl;
    std::cout << "Duration (cycles): " << complete_trace.get_duration_cycles() << std::endl;
    std::cout << "Transfer Size: " << transfer_size << " bytes" << std::endl;
    std::cout << "Bandwidth: " << payload.bandwidth_gb_s << " GB/s" << std::endl;
}

TEST_CASE_METHOD(DMATracingFixture, "Trace: Multiple DMA Transfers", "[trace][dma][queue]") {
    const size_t transfer_size = 2048;

    // Set initial cycle
    dma_engine.set_current_cycle(2000);

    size_t initial_trace_count = logger.get_trace_count();

    // Enqueue multiple transfers
    for (int i = 0; i < 3; i++) {
        auto test_data = generate_test_pattern(transfer_size, static_cast<uint8_t>(i * 0x10));
        memory_banks[0].write(i * transfer_size, test_data.data(), transfer_size);

        dma_engine.enqueue_transfer(
            DMAEngine::MemoryType::EXTERNAL, 0, i * transfer_size,
            DMAEngine::MemoryType::SCRATCHPAD, 0, i * transfer_size,
            transfer_size
        );
    }

    // Should have logged 3 issue traces
    REQUIRE(logger.get_trace_count() == initial_trace_count + 3);

    // Process all transfers
    while (dma_engine.is_busy()) {
        dma_engine.process_transfers(memory_banks, l3_tiles, l2_banks, scratchpads);
    }

    // Should have logged 3 additional completion traces (total 6 new traces)
    REQUIRE(logger.get_trace_count() == initial_trace_count + 6);

    // Get all DMA traces
    auto dma_traces = logger.get_component_traces(ComponentType::DMA_ENGINE, 0);

    // Verify all transfers have both issue and complete
    int completed_count = 0;
    for (const auto& trace : dma_traces) {
        if (trace.status == TransactionStatus::COMPLETED) {
            completed_count++;
            REQUIRE(trace.cycle_complete > trace.cycle_issue);
        }
    }

    REQUIRE(completed_count >= 3);

    std::cout << "\n=== Multiple DMA Transfers ===" << std::endl;
    std::cout << "Total traces logged: " << logger.get_trace_count() << std::endl;
    std::cout << "DMA Engine 0 traces: " << dma_traces.size() << std::endl;
}

TEST_CASE_METHOD(DMATracingFixture, "Trace: Export to CSV", "[trace][dma][export]") {
    const size_t transfer_size = 1024;

    // Generate some transfers
    dma_engine.set_current_cycle(5000);

    for (int i = 0; i < 2; i++) {
        auto test_data = generate_test_pattern(transfer_size);
        memory_banks[0].write(i * transfer_size, test_data.data(), transfer_size);

        dma_engine.enqueue_transfer(
            DMAEngine::MemoryType::EXTERNAL, 0, i * transfer_size,
            DMAEngine::MemoryType::SCRATCHPAD, 0, i * transfer_size,
            transfer_size
        );

        dma_engine.process_transfers(memory_banks, l3_tiles, l2_banks, scratchpads);
    }

    // Export traces to CSV
    bool csv_export_success = export_logger_traces("dma_trace_test.csv", "csv", logger);
    REQUIRE(csv_export_success);

    std::cout << "\n=== Trace Export ===" << std::endl;
    std::cout << "Exported " << logger.get_trace_count() << " traces to dma_trace_test.csv" << std::endl;
}

TEST_CASE_METHOD(DMATracingFixture, "Trace: Export to JSON", "[trace][dma][export]") {
    const size_t transfer_size = 1024;

    // Generate some transfers
    dma_engine.set_current_cycle(6000);

    for (int i = 0; i < 2; i++) {
        auto test_data = generate_test_pattern(transfer_size);
        memory_banks[0].write(i * transfer_size, test_data.data(), transfer_size);

        dma_engine.enqueue_transfer(
            DMAEngine::MemoryType::EXTERNAL, 0, i * transfer_size,
            DMAEngine::MemoryType::SCRATCHPAD, 0, i * transfer_size,
            transfer_size
        );

        dma_engine.process_transfers(memory_banks, l3_tiles, l2_banks, scratchpads);
    }

    // Export traces to JSON
    bool json_export_success = export_logger_traces("dma_trace_test.json", "json", logger);
    REQUIRE(json_export_success);

    std::cout << "\n=== JSON Trace Export ===" << std::endl;
    std::cout << "Exported " << logger.get_trace_count() << " traces to dma_trace_test.json" << std::endl;
}

TEST_CASE_METHOD(DMATracingFixture, "Trace: Export to Chrome Trace Format", "[trace][dma][export][chrome]") {
    const size_t transfer_size = 1024;

    // Clear previous traces for cleaner visualization
    logger.clear();

    // Generate some transfers with clear cycle progression
    for (int i = 0; i < 5; i++) {
        CycleCount start_cycle = 10000 + i * 1000;
        dma_engine.set_current_cycle(start_cycle);

        auto test_data = generate_test_pattern(transfer_size);
        memory_banks[0].write(i * transfer_size, test_data.data(), transfer_size);

        dma_engine.enqueue_transfer(
            DMAEngine::MemoryType::EXTERNAL, 0, i * transfer_size,
            DMAEngine::MemoryType::SCRATCHPAD, 0, i * transfer_size,
            transfer_size
        );

        dma_engine.process_transfers(memory_banks, l3_tiles, l2_banks, scratchpads);
    }

    // Export traces to Chrome trace format
    bool chrome_export_success = export_logger_traces("dma_trace_test.trace", "chrome", logger);
    REQUIRE(chrome_export_success);

    std::cout << "\n=== Chrome Trace Export ===" << std::endl;
    std::cout << "Exported " << logger.get_trace_count() << " traces to dma_trace_test.trace" << std::endl;
    std::cout << "Open in chrome://tracing for visualization" << std::endl;
}

TEST_CASE_METHOD(DMATracingFixture, "Trace: Cycle Range Query", "[trace][dma][query]") {
    // Clear for clean test
    logger.clear();

    // Create transfers at different cycle ranges
    std::vector<CycleCount> start_cycles = {1000, 5000, 10000, 15000};

    for (CycleCount start : start_cycles) {
        dma_engine.set_current_cycle(start);
        auto test_data = generate_test_pattern(1024);
        memory_banks[0].write(0, test_data.data(), 1024);

        dma_engine.enqueue_transfer(
            DMAEngine::MemoryType::EXTERNAL, 0, 0,
            DMAEngine::MemoryType::SCRATCHPAD, 0, 0,
            1024
        );

        dma_engine.process_transfers(memory_banks, l3_tiles, l2_banks, scratchpads);
    }

    // Query specific cycle ranges
    auto early_traces = logger.get_traces_in_range(0, 6000);
    auto late_traces = logger.get_traces_in_range(6000, 20000);

    std::cout << "\n=== Cycle Range Query ===" << std::endl;
    std::cout << "Early traces (0-6000): " << early_traces.size() << std::endl;
    std::cout << "Late traces (6000-20000): " << late_traces.size() << std::endl;

    // Should have captured traces in both ranges
    REQUIRE(early_traces.size() > 0);
    REQUIRE(late_traces.size() > 0);
}

TEST_CASE_METHOD(DMATracingFixture, "Trace: Bandwidth Analysis", "[trace][dma][analysis]") {
    // Clear for clean test
    logger.clear();

    std::vector<size_t> transfer_sizes = {1024, 4096, 16384, 65536};

    dma_engine.set_current_cycle(20000);

    for (size_t size : transfer_sizes) {
        if (size > scratchpads[0].get_capacity()) continue;

        auto test_data = generate_test_pattern(size);
        memory_banks[0].write(0, test_data.data(), size);

        dma_engine.enqueue_transfer(
            DMAEngine::MemoryType::EXTERNAL, 0, 0,
            DMAEngine::MemoryType::SCRATCHPAD, 0, 0,
            size
        );

        dma_engine.process_transfers(memory_banks, l3_tiles, l2_banks, scratchpads);
    }

    // Analyze bandwidth from traces
    auto dma_traces = logger.get_component_traces(ComponentType::DMA_ENGINE, 0);

    std::cout << "\n=== Bandwidth Analysis ===" << std::endl;
    std::cout << "Transfer Size (bytes) | Duration (cycles) | Effective BW (GB/s)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (const auto& trace : dma_traces) {
        if (trace.status == TransactionStatus::COMPLETED && std::holds_alternative<DMAPayload>(trace.payload)) {
            const auto& payload = std::get<DMAPayload>(trace.payload);
            uint64_t duration = trace.get_duration_cycles();

            if (duration > 0 && trace.clock_freq_ghz.has_value()) {
                // Effective bandwidth = bytes / (duration_cycles / clock_freq_ghz)
                double duration_ns = static_cast<double>(duration) / trace.clock_freq_ghz.value();
                double effective_bw_gb_s = (payload.bytes_transferred / duration_ns);

                std::cout << payload.bytes_transferred << " | "
                         << duration << " | "
                         << effective_bw_gb_s << std::endl;
            }
        }
    }
}
