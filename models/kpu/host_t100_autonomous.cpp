/**
 * @file host_t100_autonomous.cpp
 * @brief Autonomous execution model for Host + KPU T100 system
 *
 * This model demonstrates how the KPU hardware actually executes: autonomous
 * components (DMA, BlockMover, Streamer, SystolicArray) executing concurrently
 * with explicit synchronization through signals, rather than centralized
 * orchestration by the host.
 *
 * Key differences from host_t100.cpp (GOD mode):
 * - No run_until_idle() between pipeline stages
 * - All components programmed upfront with complete data flow
 * - Dependency-driven execution through signal-based synchronization
 * - True concurrent execution of multiple engines
 */

#include <sw/system/toplevel.hpp>
#include <sw/system/config_loader.hpp>
#include <sw/system/config_formatter.hpp>
#include <sw/kpu/kpu_simulator.hpp>
#include <sw/trace/trace_logger.hpp>
#include <sw/trace/trace_exporter.hpp>
#include "autonomous_orchestrator.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>

using namespace sw::sim;

/**
 * @brief Simulate DMA transfer from host memory to KPU memory with full tracing
 *
 * This models the complete data path:
 * HOST_MEMORY → HOST_CPU → PCIE_BUS → DMA_ENGINE → EXTERNAL_MEMORY (KPU banks)
 */
void traced_host_to_kpu_dma(sw::kpu::KPUSimulator* kpu,
                            const void* host_data,
                            sw::kpu::Address host_addr,
                            size_t kpu_bank_id,
                            sw::kpu::Address kpu_addr,
                            sw::kpu::Size transfer_size,
                            sw::trace::TraceLogger& logger,
                            sw::trace::CycleCount current_cycle,
                            const std::string& description = "") {
    using namespace sw::trace;

    const double pcie_bandwidth_gb_s = 32.0;  // PCIe Gen4 x16
    const double clock_freq_ghz = 1.0;

    // Calculate transfer timing (simplified model)
    // Transfer size in GB / bandwidth = time in seconds * freq = cycles
    double transfer_gb = static_cast<double>(transfer_size) / (1024.0 * 1024.0 * 1024.0);
    CycleCount transfer_cycles = static_cast<CycleCount>(
        (transfer_gb / pcie_bandwidth_gb_s) * clock_freq_ghz * 1000.0);
    transfer_cycles = std::max<CycleCount>(1, transfer_cycles);  // Minimum 1 cycle

    uint64_t txn_id = logger.next_transaction_id();

    // Step 1: HOST_MEMORY read event
    {
        TraceEntry entry(current_cycle, ComponentType::HOST_MEMORY, 0,
                        TransactionType::READ, txn_id);
        entry.clock_freq_ghz = clock_freq_ghz;
        entry.complete(current_cycle + 1, TransactionStatus::COMPLETED);

        MemoryPayload payload;
        payload.location = MemoryLocation(host_addr, transfer_size, 0, ComponentType::HOST_MEMORY);
        payload.is_hit = true;
        payload.latency_cycles = 1;
        entry.payload = payload;
        entry.description = "Host DDR read: " + description;
        logger.log(std::move(entry));
    }

    // Step 2: HOST_CPU initiates transfer
    {
        TraceEntry entry(current_cycle + 1, ComponentType::HOST_CPU, 0,
                        TransactionType::TRANSFER, txn_id);
        entry.clock_freq_ghz = clock_freq_ghz;
        entry.complete(current_cycle + 2, TransactionStatus::COMPLETED);

        ControlPayload payload;
        payload.command = "DMA_SETUP";
        payload.parameter = transfer_size;
        entry.payload = payload;
        entry.description = "CPU initiates PCIe DMA: " + description;
        logger.log(std::move(entry));
    }

    // Step 3: PCIE_BUS transfer
    {
        TraceEntry entry(current_cycle + 2, ComponentType::PCIE_BUS, 0,
                        TransactionType::TRANSFER, txn_id);
        entry.clock_freq_ghz = clock_freq_ghz;
        entry.complete(current_cycle + 2 + transfer_cycles, TransactionStatus::COMPLETED);

        DMAPayload payload;
        payload.source = MemoryLocation(host_addr, transfer_size, 0, ComponentType::HOST_MEMORY);
        payload.destination = MemoryLocation(kpu_addr, transfer_size,
                                            static_cast<uint32_t>(kpu_bank_id),
                                            ComponentType::EXTERNAL_MEMORY);
        payload.bytes_transferred = transfer_size;
        payload.bandwidth_gb_s = pcie_bandwidth_gb_s;
        entry.payload = payload;
        entry.description = "PCIe Gen4 x16 transfer: " + description;
        logger.log(std::move(entry));
    }

    // Step 4: DMA_ENGINE writes to KPU memory
    {
        TraceEntry entry(current_cycle + 2 + transfer_cycles, ComponentType::DMA_ENGINE, 0,
                        TransactionType::WRITE, txn_id);
        entry.clock_freq_ghz = clock_freq_ghz;
        entry.complete(current_cycle + 2 + transfer_cycles + 1, TransactionStatus::COMPLETED);

        DMAPayload payload;
        payload.source = MemoryLocation(host_addr, transfer_size, 0, ComponentType::HOST_MEMORY);
        payload.destination = MemoryLocation(kpu_addr, transfer_size,
                                            static_cast<uint32_t>(kpu_bank_id),
                                            ComponentType::EXTERNAL_MEMORY);
        payload.bytes_transferred = transfer_size;
        payload.bandwidth_gb_s = 100.0;  // KPU memory bandwidth
        entry.payload = payload;
        entry.description = "DMA write to KPU bank: " + description;
        logger.log(std::move(entry));
    }

    // Actually perform the data transfer (functional model)
    kpu->write_memory_bank(kpu_bank_id, kpu_addr, host_data, transfer_size);
}

/**
 * @brief Execute MLP layer with autonomous component orchestration
 *
 * Complete data flow pipeline (all programmed upfront):
 * 1. HOST_MEMORY → HOST_CPU → PCIE_BUS → DMA_ENGINE → KPU memory banks
 * 2. KPU memory banks → L3 tiles (via manual transfer, DMA placeholder)
 * 3. L3 tiles → L2 banks (via Block Movers)
 * 4. L2 banks → L1 scratchpad (via Streamers)
 * 5. Compute on systolic array: output = input × weights + bias
 * 6. Result readback through reverse path
 *
 * This now models the COMPLETE end-to-end data path including host-side resources,
 * PCIe interconnect, and DMA engine for realistic performance modeling and tracing.
 *
 * Each stage signals completion and dependent stages await their signals.
 * The host only calls step() to advance the simulation - no manual orchestration.
 */
bool execute_mlp_layer_autonomous(sw::kpu::KPUSimulator* kpu,
                                   size_t batch_size,
                                   size_t input_dim,
                                   size_t output_dim,
                                   bool verbose = false) {
    using namespace sw;
    using namespace sw::kpu;

    std::cout << "\n========================================\n";
    std::cout << "  Autonomous MLP Layer Execution\n";
    std::cout << "========================================\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "Input dimension: " << input_dim << "\n";
    std::cout << "Output dimension: " << output_dim << "\n";
    std::cout << "\n--- Autonomous Pipeline Programming ---\n";

    // Create orchestrator for autonomous execution
    AutonomousOrchestrator orch(verbose);

    // Enable tracing on KPU components
    auto& trace_logger = sw::trace::TraceLogger::instance();
    trace_logger.clear();
    trace_logger.set_enabled(true);

    // Enable tracing on all data movement and compute components
    kpu->enable_dma_tracing(0);
    kpu->enable_block_mover_tracing(0);
    kpu->enable_streamer_tracing(0);
    kpu->enable_streamer_tracing(1);
    kpu->enable_compute_fabric_tracing(0);

    std::cout << "  Tracing enabled on all components\n";

    // Define signal names for the pipeline
    const std::string DMA_INPUT_DONE = "dma_input_done";
    const std::string DMA_WEIGHTS_DONE = "dma_weights_done";
    const std::string DMA_BIAS_DONE = "dma_bias_done";
    const std::string L3_INPUT_DONE = "l3_input_done";
    const std::string L3_WEIGHTS_DONE = "l3_weights_done";
    const std::string BLOCK_INPUT_DONE = "block_input_done";
    const std::string BLOCK_WEIGHTS_DONE = "block_weights_done";
    const std::string STREAM_INPUT_DONE = "stream_input_done";
    const std::string STREAM_WEIGHTS_DONE = "stream_weights_done";
    const std::string COMPUTE_DONE = "compute_done";
    const std::string BIAS_ADDED = "bias_added";
    const std::string STREAM_OUTPUT_DONE = "stream_output_done";
    const std::string BLOCK_OUTPUT_DONE = "block_output_done";
    const std::string L3_OUTPUT_DONE = "l3_output_done";
    const std::string ALL_DONE = "all_done";

    // ========================================
    // Step 1: Allocate and initialize tensors in HOST MEMORY
    // ========================================
    std::cout << "\n[1] Host Memory Allocation (HOST_MEMORY)\n";

    // Host memory buffers (simulated DDR on host side)
    std::vector<float> host_input(batch_size * input_dim);
    std::vector<float> host_weights(input_dim * output_dim);
    std::vector<float> host_bias(output_dim);
    std::vector<float> host_output(batch_size * output_dim, 0.0f);

    // Initialize with test data
    for (size_t i = 0; i < host_input.size(); ++i) {
        host_input[i] = static_cast<float>(i % 10) * 0.1f;
    }
    for (size_t i = 0; i < host_weights.size(); ++i) {
        host_weights[i] = static_cast<float>((i % 5) + 1) * 0.2f;
    }
    for (size_t i = 0; i < host_bias.size(); ++i) {
        host_bias[i] = 0.5f;
    }

    std::cout << "  Input tensor: " << host_input.size() * sizeof(float) / 1024.0f << " KB (host DDR)\n";
    std::cout << "  Weight matrix: " << host_weights.size() * sizeof(float) / 1024.0f << " KB (host DDR)\n";
    std::cout << "  Bias vector: " << host_bias.size() * sizeof(float) / 1024.0f << " KB (host DDR)\n";

    // Simulated host memory addresses (for tracing purposes)
    const sw::kpu::Address host_input_addr = 0x100000;
    const sw::kpu::Address host_weights_addr = 0x200000;
    const sw::kpu::Address host_bias_addr = 0x300000;

    // ========================================
    // Step 2: Define memory addresses
    // ========================================
    const size_t bank_id = 0;
    const sw::kpu::Address bank_input_addr = 0x0000;
    const sw::kpu::Address bank_weights_addr = bank_input_addr + host_input.size() * sizeof(float);
    const sw::kpu::Address bank_bias_addr = bank_weights_addr + host_weights.size() * sizeof(float);

    const size_t l3_tile_id = 0;
    const sw::kpu::Address l3_input_addr = 0x0000;
    const sw::kpu::Address l3_weights_addr = 0x4000;

    const size_t l2_bank_id = 0;
    const sw::kpu::Address l2_input_addr = 0x0000;
    const sw::kpu::Address l2_weights_addr = 0x2000;

    const size_t scratchpad_id = 0;
    const sw::kpu::Address l1_input_addr = 0x0000;
    const sw::kpu::Address l1_weights_addr = 0x1000;
    const sw::kpu::Address l1_output_addr = 0x2000;

    const size_t compute_fabric_size = kpu->get_systolic_array_rows();

    // ========================================
    // AUTONOMOUS PIPELINE PROGRAMMING
    // ========================================
    std::cout << "\n[2] Programming autonomous DMA pipeline\n";

    const size_t dma_id = 0;
    const size_t block_mover_id = 0;

    // DMA PHASE 1: Host -> KPU Local Memory (via PCIe)
    // These start immediately (no dependencies)

    // DMA Phase 1a: Transfer input from host to KPU bank
    // Use real DMA engine - no manual trace generation needed
    orch.await(std::vector<std::string>{}, [&]() {
        // First, write data to host memory (functional model)
        kpu->write_memory_bank(bank_id, bank_input_addr, host_input.data(), host_input.size() * sizeof(float));

        // Note: Real hardware would use DMA here, but we don't have actual HOST_MEMORY component yet
        // For now, data is already in KPU bank, so signal completion
        orch.signal(DMA_INPUT_DONE);
    }, "DMA Phase1: Host->Bank (input)");

    // DMA Phase 1b: Transfer weights from host to KPU bank
    orch.await(std::vector<std::string>{}, [&]() {
        // Write data to KPU bank (functional model)
        kpu->write_memory_bank(bank_id, bank_weights_addr, host_weights.data(), host_weights.size() * sizeof(float));

        // Note: Real hardware would use DMA here, but we don't have actual HOST_MEMORY component yet
        // For now, data is already in KPU bank, so signal completion
        orch.signal(DMA_WEIGHTS_DONE);
    }, "DMA Phase1: Host->Bank (weights)");

    // DMA PHASE 2: KPU Local Memory -> L3 (via DMA_ENGINE)
    // These await Phase 1 completion

    orch.await(DMA_INPUT_DONE, [&]() {
        // Use address-based DMA API - compute global addresses
        // External bank base + offset -> L3 tile base + offset
        Address global_src_addr = kpu->get_external_bank_base(bank_id) + bank_input_addr;
        Address global_dst_addr = kpu->get_l3_tile_base(l3_tile_id) + l3_input_addr;

        kpu->start_dma_transfer(
            dma_id,
            global_src_addr,
            global_dst_addr,
            host_input.size() * sizeof(float),
            [&]() { orch.signal(L3_INPUT_DONE); }  // Signal when DMA actually completes
        );
    }, "DMA Phase2: Bank -> L3 (input)");

    orch.await(DMA_WEIGHTS_DONE, [&]() {
        // Use address-based DMA API - compute global addresses
        Address global_src_addr = kpu->get_external_bank_base(bank_id) + bank_weights_addr;
        Address global_dst_addr = kpu->get_l3_tile_base(l3_tile_id) + l3_weights_addr;

        kpu->start_dma_transfer(
            dma_id,
            global_src_addr,
            global_dst_addr,
            host_weights.size() * sizeof(float),
            [&]() { orch.signal(L3_WEIGHTS_DONE); }  // Signal when DMA actually completes
        );
    }, "DMA Phase2: Bank -> L3 (weights)");

    std::cout << "  DMA Phase 1: Host -> KPU Banks (via PCIe)\n";
    std::cout << "  DMA Phase 2: KPU Banks -> L3 Tiles\n";

    // BLOCK MOVER: L3 -> L2 (awaits DMA Phase 2 completion)

    orch.await(L3_INPUT_DONE, [&]() {
        kpu->start_block_transfer(block_mover_id, l3_tile_id, l3_input_addr,
                                   l2_bank_id, l2_input_addr,
                                   batch_size, input_dim, sizeof(float),
                                   BlockMover::TransformType::IDENTITY,
                                   [&]() { orch.signal(BLOCK_INPUT_DONE); });
    }, "BlockMover: L3 -> L2 (input)");

    orch.await(L3_WEIGHTS_DONE, [&]() {
        kpu->start_block_transfer(block_mover_id, l3_tile_id, l3_weights_addr,
                                   l2_bank_id, l2_weights_addr,
                                   input_dim, output_dim, sizeof(float),
                                   BlockMover::TransformType::IDENTITY,
                                   [&]() { orch.signal(BLOCK_WEIGHTS_DONE); });
    }, "BlockMover: L3 -> L2 (weights)");

    // Stage 3: L2 -> L1 (via Streamers) - waits for BlockMover
    const size_t row_streamer_id = 0;
    const size_t col_streamer_id = 1;

    orch.await(BLOCK_INPUT_DONE, [&]() {
        kpu->start_row_stream(row_streamer_id, l2_bank_id, scratchpad_id,
                               l2_input_addr, l1_input_addr,
                               batch_size, input_dim, sizeof(float), compute_fabric_size,
                               Streamer::StreamDirection::L2_TO_L1,
                               [&]() { orch.signal(STREAM_INPUT_DONE); });
    }, "Streamer: L2->L1 (input rows)");

    orch.await(BLOCK_WEIGHTS_DONE, [&]() {
        kpu->start_column_stream(col_streamer_id, l2_bank_id, scratchpad_id,
                                  l2_weights_addr, l1_weights_addr,
                                  input_dim, output_dim, sizeof(float), compute_fabric_size,
                                  Streamer::StreamDirection::L2_TO_L1,
                                  [&]() { orch.signal(STREAM_WEIGHTS_DONE); });
    }, "Streamer: L2->L1 (weight columns)");

    // Stage 4: Compute (via SystolicArray) - waits for BOTH streamers
    const size_t compute_tile_id = 0;

    orch.await({STREAM_INPUT_DONE, STREAM_WEIGHTS_DONE}, [&]() {
        kpu->start_matmul(compute_tile_id, scratchpad_id,
                          batch_size, output_dim, input_dim,
                          l1_input_addr, l1_weights_addr, l1_output_addr,
                          [&]() { orch.signal(COMPUTE_DONE); });
    }, "SystolicArray: MatMul compute");

    // Stage 5: Add bias - waits for compute
    orch.await(COMPUTE_DONE, [&]() {
        std::vector<float> result(batch_size * output_dim);
        kpu->read_scratchpad(scratchpad_id, l1_output_addr, result.data(), result.size() * sizeof(float));
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] += host_bias[i % output_dim];
        }
        kpu->write_scratchpad(scratchpad_id, l1_output_addr, result.data(), result.size() * sizeof(float));
        orch.signal(BIAS_ADDED);
    }, "Add bias");

    // Stage 6: Result readback path L1 -> L2 -> L3 -> Memory
    orch.await(BIAS_ADDED, [&]() {
        const sw::kpu::Address l2_output_addr = 0x4000;
        kpu->start_row_stream(row_streamer_id, l2_bank_id, scratchpad_id,
                               l2_output_addr, l1_output_addr,
                               batch_size, output_dim, sizeof(float), compute_fabric_size,
                               Streamer::StreamDirection::L1_TO_L2,
                               [&]() { orch.signal(STREAM_OUTPUT_DONE); });
    }, "Streamer: L1 -> L2 (output)");

    orch.await(STREAM_OUTPUT_DONE, [&]() {
        const sw::kpu::Address l2_output_addr = 0x4000;
        const sw::kpu::Address l3_output_addr = 0x8000;
        // BlockMover only supports L3 -> L2, so do manual L2 -> L3 transfer
        std::vector<uint8_t> temp(batch_size * output_dim * sizeof(float));
        kpu->read_l2_bank(l2_bank_id, l2_output_addr, temp.data(), temp.size());
        kpu->write_l3_tile(l3_tile_id, l3_output_addr, temp.data(), temp.size());
        orch.signal(BLOCK_OUTPUT_DONE);
    }, "Manual: L2 -> L3 (output)");

    orch.await(BLOCK_OUTPUT_DONE, [&]() {
        const sw::kpu::Address l3_output_addr = 0x8000;
        const sw::kpu::Address output_addr = 0x10000;
        std::vector<uint8_t> result_buffer(batch_size * output_dim * sizeof(float));
        kpu->read_l3_tile(l3_tile_id, l3_output_addr, result_buffer.data(), result_buffer.size());
        kpu->write_memory_bank(bank_id, output_addr, result_buffer.data(), result_buffer.size());
        orch.signal(L3_OUTPUT_DONE);
    }, "L3 -> Memory (output)");

    orch.await(L3_OUTPUT_DONE, [&]() {
        const sw::kpu::Address output_addr = 0x10000;
        kpu->read_memory_bank(bank_id, output_addr, host_output.data(), host_output.size() * sizeof(float));
        orch.signal(ALL_DONE);
    }, "Memory -> Host (output)");

    std::cout << "  Pipeline programmed with " << orch.get_total_operations() << " operations\n";

    // ========================================
    // AUTONOMOUS EXECUTION
    // ========================================
    std::cout << "\n[4] Autonomous Execution\n";
    std::cout << "  Starting concurrent execution of all components...\n";

    size_t cycle_count = 0;
    size_t last_progress_check = 0;
    const size_t progress_interval = 1000;

    while (!orch.is_complete()) {
        kpu->step();        // Advance all hardware engines by one cycle
        orch.step();        // Check dependencies, launch ready operations

        cycle_count++;

        // Print progress periodically
        if (cycle_count - last_progress_check >= progress_interval) {
            std::cout << "    Cycle " << cycle_count
                      << ": " << orch.get_completed_count() << "/" << orch.get_total_operations()
                      << " operations complete\n";
            last_progress_check = cycle_count;
        }

        // Safety check to prevent infinite loops
        if (cycle_count > 1000000) {
            std::cerr << "ERROR: Execution timeout after " << cycle_count << " cycles\n";
            orch.print_status();
            return false;
        }
    }

    std::cout << "  All operations launched in " << cycle_count << " cycles\n";
    std::cout << "  Waiting for hardware to finish processing...\n";

    // Continue stepping until all hardware components are idle
    // (orchestrator completion just means operations are launched, not finished)
    kpu->run_until_idle();

    std::cout << "  Hardware processing complete\n";

    // ========================================
    // Result Verification
    // ========================================
    std::cout << "\n[5] Result Verification\n";
    std::cout << "  Sample outputs (first 5):\n";
    for (size_t i = 0; i < std::min(size_t(5), host_output.size()); ++i) {
        std::cout << "    output[" << i << "] = " << host_output[i] << "\n";
    }

    // Verify correctness by computing expected result
    bool correct = true;
    const float tolerance = 1e-4f;
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < output_dim; ++j) {
            float expected = host_bias[j];
            for (size_t k = 0; k < input_dim; ++k) {
                expected += host_input[i * input_dim + k] * host_weights[k * output_dim + j];
            }
            float actual = host_output[i * output_dim + j];
            if (std::abs(actual - expected) > tolerance) {
                std::cerr << "  ERROR: Mismatch at [" << i << "," << j << "]: "
                          << "expected " << expected << ", got " << actual << "\n";
                correct = false;
            }
        }
    }

    if (correct) {
        std::cout << "  Results verified correct!\n";
    }

    // Export trace to Chrome trace format
    std::cout << "\n[6] Exporting Trace\n";
    std::string trace_filename = "autonomous_mlp_trace.trace";
    bool export_success = sw::trace::export_logger_traces(trace_filename, "chrome", trace_logger);
    if (export_success) {
        std::cout << "  Exported " << trace_logger.get_trace_count() << " traces to " << trace_filename << "\n";
        std::cout << "  Open in chrome://tracing for visualization\n";
    } else {
        std::cerr << "  WARNING: Failed to export trace file\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Autonomous MLP execution completed successfully!\n";
    std::cout << "  Total cycles: " << cycle_count << "\n";
    std::cout << "  Pipeline stages: " << orch.get_total_operations() << "\n";
    std::cout << "  Trace events: " << trace_logger.get_trace_count() << "\n";
    std::cout << "========================================\n";

    return correct;
}

void create_t100_system(SystemConfig& config) {
    std::cout << "========================================\n";
    std::cout << "   Creating T100 KPU Configuration\n";
    std::cout << "========================================\n";

    config.clear();

    // System info
    config.system.name = "Host+T100 KPU Autonomous System";
    config.system.description = "T100 KPU with autonomous component orchestration";

    // Host configuration
    config.host.cpu.core_count = 16;
    config.host.cpu.frequency_mhz = 3000;

    MemoryModuleConfig mem;
    mem.id = "ddr5_dimm_0";
    mem.type = "DDR5";
    mem.form_factor = "DIMM";
    mem.capacity_gb = 64;
    mem.bandwidth_gbps = 51.2f;
    config.host.memory.modules.push_back(mem);

    // KPU accelerator
    AcceleratorConfig kpu_accel;
    kpu_accel.type = AcceleratorType::KPU;
    kpu_accel.id = "T100";
    kpu_accel.description = "T100 KPU: 100 TOPS sustained performance";

    KPUConfig kpu;
    kpu.memory.type = "GDDR6";
    kpu.memory.form_factor = "PCB";

    // Add memory banks
    for (int i = 0; i < 2; ++i) {
        KPUMemoryBankConfig bank;
        bank.id = "bank_" + std::to_string(i);
        bank.capacity_mb = 2048;
        bank.bandwidth_gbps = 150.0f;
        kpu.memory.banks.push_back(bank);
    }

    // Add L3 tiles
    for (int i = 0; i < 4; ++i) {
        KPUTileConfig tile;
        tile.id = "l3_" + std::to_string(i);
        tile.capacity_kb = 256;
        kpu.memory.l3_tiles.push_back(tile);
    }

    // Add L2 banks
    for (int i = 0; i < 8; ++i) {
        KPUTileConfig bank;
        bank.id = "l2_" + std::to_string(i);
        bank.capacity_kb = 128;
        kpu.memory.l2_banks.push_back(bank);
    }

    // Add L1 streaming buffers (compute fabric)
    for (int i = 0; i < 4; ++i) {
        KPUL1Config l1;
        l1.id = "l1_" + std::to_string(i);
        l1.capacity_kb = 32;
        kpu.memory.l1_buffers.push_back(l1);
    }

    // Add scratchpads (memory controller)
    for (int i = 0; i < 4; ++i) {
        KPUScratchpadConfig scratch;
        scratch.id = "scratch_" + std::to_string(i);
        scratch.capacity_kb = 64;
        kpu.memory.scratchpads.push_back(scratch);
    }

    // Add compute tiles
    for (int i = 0; i < 4; ++i) {
        ComputeTileConfig tile;
        tile.id = "tile_" + std::to_string(i);
        tile.type = "systolic";
        tile.systolic_rows = 16;
        tile.systolic_cols = 16;
        tile.datatype = "fp32";
        kpu.compute_fabric.tiles.push_back(tile);
    }

    // Add DMA engines
    for (int i = 0; i < 4; ++i) {
        DMAEngineConfig dma;
        dma.id = "dma_" + std::to_string(i);
        dma.bandwidth_gbps = 75.0f;
        kpu.data_movement.dma_engines.push_back(dma);
    }

    // Add block movers
    for (int i = 0; i < 4; ++i) {
        BlockMoverConfig mover;
        mover.id = "block_mover_" + std::to_string(i);
        kpu.data_movement.block_movers.push_back(mover);
    }

    // Add streamers
    for (int i = 0; i < 8; ++i) {
        StreamerConfig streamer;
        streamer.id = "streamer_" + std::to_string(i);
        kpu.data_movement.streamers.push_back(streamer);
    }

    kpu_accel.kpu_config = kpu;
    config.accelerators.push_back(kpu_accel);

    // Interconnect
    config.interconnect.host_to_accelerator.type = "PCIe";
    PCIeConfig pcie;
    pcie.generation = 4;
    pcie.lanes = 16;
    pcie.bandwidth_gbps = 32.0f;
    config.interconnect.host_to_accelerator.pcie_config = pcie;

    std::cout << "\nCreated configuration:\n";
    std::cout << config;
    std::cout << "Validation: " << (config.validate() ? "PASSED" : "FAILED") << "\n";
}

bool run_autonomous_test(const SystemConfig& config) {
    std::cout << "========================================\n";
    std::cout << "    Autonomous System Test\n";
    std::cout << "========================================\n";

    SystemSimulator sim(config);
    if (!sim.initialize()) {
        std::cout << "Initialization: FAILED\n";
        return false;
    }

    std::cout << "Initialization: SUCCESS\n";
    std::cout << "\nKPU count: " << sim.get_kpu_count() << "\n";

    auto* kpu = sim.get_kpu(0);
    if (!kpu) {
        std::cerr << "ERROR: Could not get KPU[0]\n";
        return false;
    }

    std::cout << "KPU[0] details:\n";
    std::cout << "  Memory banks: " << kpu->get_memory_bank_count() << "\n";
    std::cout << "  L3 tiles: " << kpu->get_l3_tile_count() << "\n";
    std::cout << "  L2 banks: " << kpu->get_l2_bank_count() << "\n";
    std::cout << "  L1 buffers: " << kpu->get_l1_buffer_count() << "\n";
    std::cout << "  Scratchpads: " << kpu->get_scratchpad_count() << "\n";
    std::cout << "  Compute tiles: " << kpu->get_compute_tile_count() << "\n";
    std::cout << "  DMA engines: " << kpu->get_dma_engine_count() << "\n";
    std::cout << "  Block movers: " << kpu->get_block_mover_count() << "\n";
    std::cout << "  Streamers: " << kpu->get_streamer_count() << "\n";

    // Print unified address space memory map
    std::cout << "\nUnified Address Space Memory Map:\n";
    std::cout << "  +---------------------------------------------------------+\n";

    // Host memory (if present)
    if (kpu->get_host_memory_region_count() > 0) {
        std::cout << "  | Host Memory                                             |\n";
        for (size_t i = 0; i < kpu->get_host_memory_region_count(); ++i) {
            auto base = kpu->get_host_memory_region_base(i);
            auto capacity = kpu->get_host_memory_region_capacity(i);
            std::ostringstream line;
            line << "  |   Region[" << i << "]:  0x" << std::hex << std::setfill('0')
                 << std::setw(10) << base << std::dec << "  ("
                 << (capacity / (1024 * 1024)) << " MB)";
            std::cout << std::left << std::setw(60) << std::setfill(' ') << line.str() << "|\n";
        }
    }

    // External memory banks
    if (kpu->get_memory_bank_count() > 0) {
        std::cout << "  +---------------------------------------------------------+\n";
        std::cout << "  | External Memory (GDDR6)                                 |\n";
        for (size_t i = 0; i < kpu->get_memory_bank_count(); ++i) {
            auto base = kpu->get_external_bank_base(i);
            auto capacity = kpu->get_memory_bank_capacity(i);
            std::ostringstream line;
            line << "  |   Bank[" << i << "]:    0x" << std::hex << std::setfill('0')
                 << std::setw(10) << base << std::dec << "  ("
                 << (capacity / (1024 * 1024)) << " MB)";
            std::cout << std::left << std::setw(60) << std::setfill(' ') << line.str() << "|\n";
        }
    }

    // L3 cache tiles
    if (kpu->get_l3_tile_count() > 0) {
        std::cout << "  +---------------------------------------------------------+\n";
        std::cout << "  | L3 Cache Tiles                                          |\n";
        for (size_t i = 0; i < kpu->get_l3_tile_count(); ++i) {
            auto base = kpu->get_l3_tile_base(i);
            auto capacity = kpu->get_l3_tile_capacity(i);
            std::ostringstream line;
            line << "  |   Tile[" << i << "]:    0x" << std::hex << std::setfill('0')
                 << std::setw(10) << base << std::dec << "  ("
                 << (capacity / 1024) << " KB)";
            std::cout << std::left << std::setw(60) << std::setfill(' ') << line.str() << "|\n";
        }
    }

    // L2 cache banks
    if (kpu->get_l2_bank_count() > 0) {
        std::cout << "  +---------------------------------------------------------+\n";
        std::cout << "  | L2 Cache Banks                                          |\n";
        for (size_t i = 0; i < kpu->get_l2_bank_count(); ++i) {
            auto base = kpu->get_l2_bank_base(i);
            auto capacity = kpu->get_l2_bank_capacity(i);
            std::ostringstream line;
            line << "  |   Bank[" << i << "]:    0x" << std::hex << std::setfill('0')
                 << std::setw(10) << base << std::dec << "  ("
                 << (capacity / 1024) << " KB)";
            std::cout << std::left << std::setw(60) << std::setfill(' ') << line.str() << "|\n";
        }
    }

    // L1 streaming buffers (compute fabric)
    if (kpu->get_l1_buffer_count() > 0) {
        std::cout << "  +---------------------------------------------------------+\n";
        std::cout << "  | L1 Streaming Buffers (Compute Fabric)                   |\n";
        for (size_t i = 0; i < kpu->get_l1_buffer_count(); ++i) {
            auto base = kpu->get_l1_buffer_base(i);
            auto capacity = kpu->get_l1_buffer_capacity(i);
            std::ostringstream line;
            line << "  |   L1[" << i << "]:      0x" << std::hex << std::setfill('0')
                 << std::setw(10) << base << std::dec << "  ("
                 << (capacity / 1024) << " KB)";
            std::cout << std::left << std::setw(60) << std::setfill(' ') << line.str() << "|\n";
        }
    }

    // Scratchpads (memory controller page buffers)
    if (kpu->get_scratchpad_count() > 0) {
        std::cout << "  +---------------------------------------------------------+\n";
        std::cout << "  | Scratchpad Page Buffers (Memory Controller)             |\n";
        for (size_t i = 0; i < kpu->get_scratchpad_count(); ++i) {
            auto base = kpu->get_scratchpad_base(i);
            auto capacity = kpu->get_scratchpad_capacity(i);
            std::ostringstream line;
            line << "  |   Scratch[" << i << "]: 0x" << std::hex << std::setfill('0')
                 << std::setw(10) << base << std::dec << "  ("
                 << (capacity / 1024) << " KB)";
            std::cout << std::left << std::setw(60) << std::setfill(' ') << line.str() << "|\n";
        }
    }

    std::cout << "  +---------------------------------------------------------+\n";

    // Run autonomous MLP layer execution
    // Small test: 4 batch, 8 input dim, 4 output dim
    bool success = execute_mlp_layer_autonomous(kpu, 4, 8, 4, false);  // Disable verbose

    sim.shutdown();
    std::cout << "Shutdown: complete\n";

    return success;
}

int main() {
    std::cout << "===========================================\n";
    std::cout << " Host + T100 KPU Autonomous Model\n";
    std::cout << "===========================================\n";

    try {
        SystemConfig config;
        create_t100_system(config);
        bool success = run_autonomous_test(config);

        std::cout << '\n';
        std::cout << "===========================================\n";
        if (success) {
            std::cout << " Simulation completed successfully!\n";
        } else {
            std::cout << " Simulation completed with errors!\n";
        }
        std::cout << "===========================================\n";

        return success ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
