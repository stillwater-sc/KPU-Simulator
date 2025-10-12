#include <algorithm>
#include <random>
#include <iostream>
#include <cassert>
#include <cstring>

#include "sw/kpu/kpu_simulator.hpp"

namespace sw::kpu {

// KPUSimulator implementation - clean delegation-based API
KPUSimulator::KPUSimulator(const Config& config) : current_cycle(0) {
    // Initialize memory banks
    memory_banks.reserve(config.memory_bank_count);
    for (size_t i = 0; i < config.memory_bank_count; ++i) {
        memory_banks.emplace_back(config.memory_bank_capacity_mb, config.memory_bandwidth_gbps);
    }

    // Initialize scratchpads
    scratchpads.reserve(config.scratchpad_count);
    for (size_t i = 0; i < config.scratchpad_count; ++i) {
        scratchpads.emplace_back(config.scratchpad_capacity_kb);
    }

    // Initialize compute tiles with systolic array configuration
    compute_tiles.reserve(config.compute_tile_count);
    for (size_t i = 0; i < config.compute_tile_count; ++i) {
        ComputeFabric::ComputeType compute_type = config.use_systolic_arrays ?
            ComputeFabric::ComputeType::SYSTOLIC_ARRAY :
            ComputeFabric::ComputeType::BASIC_MATMUL;

        compute_tiles.emplace_back(i, compute_type,
                                  config.systolic_array_rows,
                                  config.systolic_array_cols);
    }

    // Initialize DMA engines - now bidirectional, configured per-transfer
    dma_engines.reserve(config.dma_engine_count);
    for (size_t i = 0; i < config.dma_engine_count; ++i) {
        dma_engines.emplace_back(i);  // Just pass engine ID for identification
    }

    // Initialize L3 tiles - distributed L3 cache tiles
    l3_tiles.reserve(config.l3_tile_count);
    for (size_t i = 0; i < config.l3_tile_count; ++i) {
        l3_tiles.emplace_back(i, config.l3_tile_capacity_kb);
    }

    // Initialize L2 banks - L2 cache banks
    l2_banks.reserve(config.l2_bank_count);
    for (size_t i = 0; i < config.l2_bank_count; ++i) {
        l2_banks.emplace_back(i, config.l2_bank_capacity_kb);
    }

    // Initialize BlockMovers - L3↔L2 data movement engines
    block_movers.reserve(config.block_mover_count);
    for (size_t i = 0; i < config.block_mover_count; ++i) {
        // Associate each BlockMover with an L3 tile (simple 1:1 mapping for now)
        size_t associated_l3_tile = i % config.l3_tile_count;
        block_movers.emplace_back(i, associated_l3_tile);
    }

    // Initialize Streamers - L2↔L1 data movement for systolic arrays
    streamers.reserve(config.streamer_count);
    for (size_t i = 0; i < config.streamer_count; ++i) {
        streamers.emplace_back(i);
    }

    sim_start_time = std::chrono::high_resolution_clock::now();
}

// Memory operations - clean delegation
void KPUSimulator::read_memory_bank(size_t bank_id, Address addr, void* data, Size size) {
    validate_bank_id(bank_id);
    memory_banks[bank_id].read(addr, data, size);
}

void KPUSimulator::write_memory_bank(size_t bank_id, Address addr, const void* data, Size size) {
    validate_bank_id(bank_id);
    memory_banks[bank_id].write(addr, data, size);
}

void KPUSimulator::read_scratchpad(size_t pad_id, Address addr, void* data, Size size) {
    validate_scratchpad_id(pad_id);
    scratchpads[pad_id].read(addr, data, size);
}

void KPUSimulator::write_scratchpad(size_t pad_id, Address addr, const void* data, Size size) {
    validate_scratchpad_id(pad_id);
    scratchpads[pad_id].write(addr, data, size);
}

// L3 and L2 memory operations
void KPUSimulator::read_l3_tile(size_t tile_id, Address addr, void* data, Size size) {
    validate_l3_tile_id(tile_id);
    l3_tiles[tile_id].read(addr, data, size);
}

void KPUSimulator::write_l3_tile(size_t tile_id, Address addr, const void* data, Size size) {
    validate_l3_tile_id(tile_id);
    l3_tiles[tile_id].write(addr, data, size);
}

void KPUSimulator::read_l2_bank(size_t bank_id, Address addr, void* data, Size size) {
    validate_l2_bank_id(bank_id);
    l2_banks[bank_id].read(addr, data, size);
}

void KPUSimulator::write_l2_bank(size_t bank_id, Address addr, const void* data, Size size) {
    validate_l2_bank_id(bank_id);
    l2_banks[bank_id].write(addr, data, size);
}

// DMA operations
void KPUSimulator::start_dma_transfer(size_t dma_id,
                                     DMAEngine::MemoryType src_type, size_t src_id, Address src_addr,
                                     DMAEngine::MemoryType dst_type, size_t dst_id, Address dst_addr,
                                     Size size, std::function<void()> callback) {
    validate_dma_id(dma_id);
    dma_engines[dma_id].enqueue_transfer(src_type, src_id, src_addr,
                                        dst_type, dst_id, dst_addr,
                                        size, std::move(callback));
}

void KPUSimulator::start_dma_external_to_scratchpad(size_t dma_id, size_t bank_id, Address src_addr,
                                                    size_t pad_id, Address dst_addr, Size size,
                                                    std::function<void()> callback) {
    start_dma_transfer(dma_id, DMAEngine::MemoryType::EXTERNAL, bank_id, src_addr,
                      DMAEngine::MemoryType::SCRATCHPAD, pad_id, dst_addr,
                      size, std::move(callback));
}

void KPUSimulator::start_dma_scratchpad_to_external(size_t dma_id, size_t pad_id, Address src_addr,
                                                    size_t bank_id, Address dst_addr, Size size,
                                                    std::function<void()> callback) {
    start_dma_transfer(dma_id, DMAEngine::MemoryType::SCRATCHPAD, pad_id, src_addr,
                      DMAEngine::MemoryType::EXTERNAL, bank_id, dst_addr,
                      size, std::move(callback));
}

bool KPUSimulator::is_dma_busy(size_t dma_id) {
    validate_dma_id(dma_id);
    return dma_engines[dma_id].is_busy();
}

// BlockMover operations
void KPUSimulator::start_block_transfer(size_t block_mover_id, size_t src_l3_tile_id, Address src_offset,
                                       size_t dst_l2_bank_id, Address dst_offset,
                                       Size block_height, Size block_width, Size element_size,
                                       BlockMover::TransformType transform,
                                       std::function<void()> callback) {
    validate_block_mover_id(block_mover_id);
    validate_l3_tile_id(src_l3_tile_id);
    validate_l2_bank_id(dst_l2_bank_id);

    block_movers[block_mover_id].enqueue_block_transfer(
        src_l3_tile_id, src_offset, dst_l2_bank_id, dst_offset,
        block_height, block_width, element_size, transform, std::move(callback)
    );
}

bool KPUSimulator::is_block_mover_busy(size_t block_mover_id) {
    validate_block_mover_id(block_mover_id);
    return block_movers[block_mover_id].is_busy();
}

// Streamer operations
void KPUSimulator::start_row_stream(size_t streamer_id, size_t l2_bank_id, size_t l1_scratchpad_id,
                                   Address l2_base_addr, Address l1_base_addr,
                                   Size matrix_height, Size matrix_width, Size element_size, Size compute_fabric_size,
                                   Streamer::StreamDirection direction,
                                   std::function<void()> callback) {
    validate_streamer_id(streamer_id);
    validate_l2_bank_id(l2_bank_id);
    validate_scratchpad_id(l1_scratchpad_id);

    Streamer::StreamConfig config{
        .l2_bank_id = l2_bank_id,
        .l1_scratchpad_id = l1_scratchpad_id,
        .l2_base_addr = l2_base_addr,
        .l1_base_addr = l1_base_addr,
        .matrix_height = matrix_height,
        .matrix_width = matrix_width,
        .element_size = element_size,
        .compute_fabric_size = compute_fabric_size,
        .direction = direction,
        .stream_type = Streamer::StreamType::ROW_STREAM,
        .cache_line_size = 64, // Default cache line size
        .completion_callback = std::move(callback)
    };

    streamers[streamer_id].enqueue_stream(config);
}

void KPUSimulator::start_column_stream(size_t streamer_id, size_t l2_bank_id, size_t l1_scratchpad_id,
                                      Address l2_base_addr, Address l1_base_addr,
                                      Size matrix_height, Size matrix_width, Size element_size, Size compute_fabric_size,
                                      Streamer::StreamDirection direction,
                                      std::function<void()> callback) {
    validate_streamer_id(streamer_id);
    validate_l2_bank_id(l2_bank_id);
    validate_scratchpad_id(l1_scratchpad_id);

    Streamer::StreamConfig config{
        .l2_bank_id = l2_bank_id,
        .l1_scratchpad_id = l1_scratchpad_id,
        .l2_base_addr = l2_base_addr,
        .l1_base_addr = l1_base_addr,
        .matrix_height = matrix_height,
        .matrix_width = matrix_width,
        .element_size = element_size,
        .compute_fabric_size = compute_fabric_size,
        .direction = direction,
        .stream_type = Streamer::StreamType::COLUMN_STREAM,
        .cache_line_size = 64, // Default cache line size
        .completion_callback = std::move(callback)
    };

    streamers[streamer_id].enqueue_stream(config);
}

bool KPUSimulator::is_streamer_busy(size_t streamer_id) {
    validate_streamer_id(streamer_id);
    return streamers[streamer_id].is_busy();
}

// Compute operations
void KPUSimulator::start_matmul(size_t tile_id, size_t scratchpad_id, Size m, Size n, Size k,
                               Address a_addr, Address b_addr, Address c_addr,
                               std::function<void()> callback) {
    validate_tile_id(tile_id);
    validate_scratchpad_id(scratchpad_id);

    ComputeFabric::MatMulConfig config{
        .m = m, .n = n, .k = k,
        .a_addr = a_addr, .b_addr = b_addr, .c_addr = c_addr,
        .scratchpad_id = scratchpad_id,
        .completion_callback = std::move(callback)
    };

    compute_tiles[tile_id].start_matmul(config);
}

bool KPUSimulator::is_compute_busy(size_t tile_id) {
    validate_tile_id(tile_id);
    return compute_tiles[tile_id].is_busy();
}

// Simulation control
void KPUSimulator::reset() {
    for (auto& bank : memory_banks) {
        bank.reset();
    }
    for (auto& pad : scratchpads) {
        pad.reset();
    }
    for (auto& dma : dma_engines) {
        dma.reset();
    }
    for (auto& tile : compute_tiles) {
        tile.reset();
    }
    for (auto& l3_tile : l3_tiles) {
        l3_tile.reset();
    }
    for (auto& l2_bank : l2_banks) {
        l2_bank.reset();
    }
    for (auto& block_mover : block_movers) {
        block_mover.reset();
    }
    for (auto& streamer : streamers) {
        streamer.reset();
    }
    current_cycle = 0;
    sim_start_time = std::chrono::high_resolution_clock::now();
}

void KPUSimulator::step() {
    ++current_cycle;

    // Update cycle on all components FIRST so they know current cycle when operations are enqueued
    for (auto& dma : dma_engines) {
        dma.set_current_cycle(current_cycle);
    }
    for (auto& block_mover : block_movers) {
        block_mover.set_cycle(current_cycle);
    }
    for (auto& streamer : streamers) {
        streamer.set_cycle(current_cycle);
    }
    for (auto& tile : compute_tiles) {
        tile.set_cycle(current_cycle);
    }

    // Then process/update all components
    for (auto& dma : dma_engines) {
        dma.process_transfers(memory_banks, l3_tiles, l2_banks, scratchpads);
    }
    for (auto& block_mover : block_movers) {
        block_mover.process_transfers(l3_tiles, l2_banks);
    }
    for (auto& streamer : streamers) {
        streamer.update(current_cycle, l2_banks, scratchpads);
    }
    for (auto& tile : compute_tiles) {
        tile.update(current_cycle, scratchpads);
    }
}

void KPUSimulator::run_until_idle() {
    bool any_busy;
    do {
        any_busy = false;
        for (const auto& dma : dma_engines) {
            if (dma.is_busy()) {
                any_busy = true;
                break;
            }
        }
        for (const auto& block_mover : block_movers) {
            if (block_mover.is_busy()) {
                any_busy = true;
                break;
            }
        }
        for (const auto& streamer : streamers) {
            if (streamer.is_busy()) {
                any_busy = true;
                break;
            }
        }
        for (const auto& tile : compute_tiles) {
            if (tile.is_busy()) {
                any_busy = true;
                break;
            }
        }
        if (any_busy) {
            step();
        }
    } while (any_busy);
}

// Configuration queries
Size KPUSimulator::get_memory_bank_capacity(size_t bank_id) const {
    validate_bank_id(bank_id);
    return memory_banks[bank_id].get_capacity();
}

Size KPUSimulator::get_scratchpad_capacity(size_t pad_id) const {
    validate_scratchpad_id(pad_id);
    return scratchpads[pad_id].get_capacity();
}

Size KPUSimulator::get_l3_tile_capacity(size_t tile_id) const {
    validate_l3_tile_id(tile_id);
    return l3_tiles[tile_id].get_capacity();
}

Size KPUSimulator::get_l2_bank_capacity(size_t bank_id) const {
    validate_l2_bank_id(bank_id);
    return l2_banks[bank_id].get_capacity();
}

// High-level test operation
bool KPUSimulator::run_matmul_test(const MatMulTest& test, size_t memory_bank_id,
                                  size_t scratchpad_id, size_t compute_tile_id) {
    reset();

    Size a_size = test.m * test.k * sizeof(float);
    Size b_size = test.k * test.n * sizeof(float);
    Size c_size = test.m * test.n * sizeof(float);

    // Addresses in external memory
    Address ext_a_addr = 0;
    Address ext_b_addr = a_size;
    Address ext_c_addr = ext_b_addr + b_size;

    // Addresses in scratchpad
    Address scratch_a_addr = 0;
    Address scratch_b_addr = a_size;
    Address scratch_c_addr = scratch_b_addr + b_size;

    try {
        // Load test data into external memory
        write_memory_bank(memory_bank_id, ext_a_addr, test.matrix_a.data(), a_size);
        write_memory_bank(memory_bank_id, ext_b_addr, test.matrix_b.data(), b_size);

        // Set up computation pipeline
        bool dma_a_complete = false, dma_b_complete = false, compute_complete = false;

        // DMA A and B matrices to scratchpad using convenience methods
        start_dma_external_to_scratchpad(0, memory_bank_id, ext_a_addr, scratchpad_id, scratch_a_addr, a_size,
            [&dma_a_complete]() { dma_a_complete = true; });
        start_dma_external_to_scratchpad(0, memory_bank_id, ext_b_addr, scratchpad_id, scratch_b_addr, b_size,
            [&dma_b_complete]() { dma_b_complete = true; });

        // Wait for data to be loaded
        while (!dma_a_complete || !dma_b_complete) {
            step();
        }

        // Start matrix multiplication
        start_matmul(compute_tile_id, scratchpad_id, test.m, test.n, test.k,
                    scratch_a_addr, scratch_b_addr, scratch_c_addr,
                    [&compute_complete]() { compute_complete = true; });

        // Wait for computation to complete
        while (!compute_complete) {
            step();
        }

        // DMA result back to external memory using convenience method
        bool dma_c_complete = false;
        start_dma_scratchpad_to_external(0, scratchpad_id, scratch_c_addr, memory_bank_id, ext_c_addr, c_size,
            [&dma_c_complete]() { dma_c_complete = true; });

        // Wait for result transfer
        while (!dma_c_complete) {
            step();
        }

        // Verify result
        std::vector<float> result_c(test.m * test.n);
        read_memory_bank(memory_bank_id, ext_c_addr, result_c.data(), c_size);

        return test_utils::verify_matmul_result(test.matrix_a, test.matrix_b, result_c,
                                               test.m, test.n, test.k);
    }
    catch (const std::exception& e) {
        std::cerr << "Error during matmul test: " << e.what() << std::endl;
        return false;
    }
}

// Statistics and monitoring
double KPUSimulator::get_elapsed_time_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - sim_start_time);
    return duration.count() / 1000.0;
}

void KPUSimulator::print_stats() const {
    std::cout << "=== KPU Simulator Statistics ===" << std::endl;
    std::cout << "Simulation cycles: " << current_cycle << std::endl;
    std::cout << "Wall-clock time: " << get_elapsed_time_ms() << " ms" << std::endl;
    std::cout << "Memory banks: " << memory_banks.size() << std::endl;
    std::cout << "Scratchpads: " << scratchpads.size() << std::endl;
    std::cout << "L3 tiles: " << l3_tiles.size() << std::endl;
    std::cout << "L2 banks: " << l2_banks.size() << std::endl;
    std::cout << "Compute tiles: " << compute_tiles.size() << std::endl;
    std::cout << "DMA engines: " << dma_engines.size() << std::endl;
    std::cout << "Block movers: " << block_movers.size() << std::endl;
}

void KPUSimulator::print_component_status() const {
    std::cout << "=== Component Status ===" << std::endl;

    std::cout << "Memory Banks:" << std::endl;
    for (size_t i = 0; i < memory_banks.size(); ++i) {
        std::cout << "  Bank[" << i << "]: " << memory_banks[i].get_capacity() / (1024*1024)
                  << " MB, Ready: " << (memory_banks[i].is_ready() ? "Yes" : "No") << std::endl;
    }

    std::cout << "Scratchpads:" << std::endl;
    for (size_t i = 0; i < scratchpads.size(); ++i) {
        std::cout << "  Pad[" << i << "]: " << scratchpads[i].get_capacity() / 1024
                  << " KB, Ready: " << (scratchpads[i].is_ready() ? "Yes" : "No") << std::endl;
    }

    std::cout << "DMA Engines:" << std::endl;
    for (size_t i = 0; i < dma_engines.size(); ++i) {
        const auto& dma = dma_engines[i];
        std::cout << "  DMA[" << i << "]: ";
        std::cout << "Busy: " << (dma.is_busy() ? "Yes" : "No");
        std::cout << ", Queue: " << dma.get_queue_size() << " transfers" << std::endl;
    }

    std::cout << "L3 Tiles:" << std::endl;
    for (size_t i = 0; i < l3_tiles.size(); ++i) {
        std::cout << "  L3Tile[" << i << "]: " << l3_tiles[i].get_capacity() / 1024
                  << " KB, Ready: " << (l3_tiles[i].is_ready() ? "Yes" : "No") << std::endl;
    }

    std::cout << "L2 Banks:" << std::endl;
    for (size_t i = 0; i < l2_banks.size(); ++i) {
        std::cout << "  L2Bank[" << i << "]: " << l2_banks[i].get_capacity() / 1024
                  << " KB, Ready: " << (l2_banks[i].is_ready() ? "Yes" : "No") << std::endl;
    }

    std::cout << "Block Movers:" << std::endl;
    for (size_t i = 0; i < block_movers.size(); ++i) {
        const auto& mover = block_movers[i];
        std::cout << "  BlockMover[" << i << "]: ";
        std::cout << "Busy: " << (mover.is_busy() ? "Yes" : "No");
        std::cout << ", Queue: " << mover.get_queue_size() << " transfers";
        std::cout << ", Associated L3: " << mover.get_associated_l3_tile() << std::endl;
    }

    std::cout << "Compute Tiles:" << std::endl;
    for (size_t i = 0; i < compute_tiles.size(); ++i) {
        std::cout << "  Tile[" << i << "]: Busy: "
                  << (compute_tiles[i].is_busy() ? "Yes" : "No") << std::endl;
    }
}

// Component status queries
bool KPUSimulator::is_memory_bank_ready(size_t bank_id) const {
    validate_bank_id(bank_id);
    return memory_banks[bank_id].is_ready();
}

bool KPUSimulator::is_scratchpad_ready(size_t pad_id) const {
    validate_scratchpad_id(pad_id);
    return scratchpads[pad_id].is_ready();
}

bool KPUSimulator::is_l3_tile_ready(size_t tile_id) const {
    validate_l3_tile_id(tile_id);
    return l3_tiles[tile_id].is_ready();
}

bool KPUSimulator::is_l2_bank_ready(size_t bank_id) const {
    validate_l2_bank_id(bank_id);
    return l2_banks[bank_id].is_ready();
}

// Validation helpers
void KPUSimulator::validate_bank_id(size_t bank_id) const {
    if (bank_id >= memory_banks.size()) {
        throw std::out_of_range("Invalid memory bank ID: " + std::to_string(bank_id));
    }
}

void KPUSimulator::validate_scratchpad_id(size_t pad_id) const {
    if (pad_id >= scratchpads.size()) {
        throw std::out_of_range("Invalid scratchpad ID: " + std::to_string(pad_id));
    }
}

void KPUSimulator::validate_dma_id(size_t dma_id) const {
    if (dma_id >= dma_engines.size()) {
        throw std::out_of_range("Invalid DMA engine ID: " + std::to_string(dma_id));
    }
}

void KPUSimulator::validate_tile_id(size_t tile_id) const {
    if (tile_id >= compute_tiles.size()) {
        throw std::out_of_range("Invalid compute tile ID: " + std::to_string(tile_id));
    }
}

void KPUSimulator::validate_l3_tile_id(size_t tile_id) const {
    if (tile_id >= l3_tiles.size()) {
        throw std::out_of_range("Invalid L3 tile ID: " + std::to_string(tile_id));
    }
}

void KPUSimulator::validate_l2_bank_id(size_t bank_id) const {
    if (bank_id >= l2_banks.size()) {
        throw std::out_of_range("Invalid L2 bank ID: " + std::to_string(bank_id));
    }
}

void KPUSimulator::validate_block_mover_id(size_t mover_id) const {
    if (mover_id >= block_movers.size()) {
        throw std::out_of_range("Invalid block mover ID: " + std::to_string(mover_id));
    }
}

void KPUSimulator::validate_streamer_id(size_t streamer_id) const {
    if (streamer_id >= streamers.size()) {
        throw std::out_of_range("Invalid streamer ID: " + std::to_string(streamer_id));
    }
}

// Test utilities implementation
namespace test_utils {

KPUSimulator::MatMulTest generate_simple_matmul_test(Size m, Size n, Size k) {
    KPUSimulator::MatMulTest test;
    test.m = m;
    test.n = n;
    test.k = k;

    test.matrix_a = generate_random_matrix(m, k, -2.0f, 2.0f);
    test.matrix_b = generate_random_matrix(k, n, -2.0f, 2.0f);

    // Compute expected result
    test.expected_c.resize(m * n);
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (Size ki = 0; ki < k; ++ki) {
                sum += test.matrix_a[i * k + ki] * test.matrix_b[ki * n + j];
            }
            test.expected_c[i * n + j] = sum;
        }
    }

    return test;
}

std::vector<float> generate_random_matrix(Size rows, Size cols, float min_val, float max_val) {
    std::vector<float> matrix(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    std::generate(matrix.begin(), matrix.end(), [&]() { return dis(gen); });
    return matrix;
}

bool verify_matmul_result(const std::vector<float>& a, const std::vector<float>& b,
                         const std::vector<float>& c, Size m, Size n, Size k, float tolerance) {
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            float expected = 0.0f;
            for (Size ki = 0; ki < k; ++ki) {
                expected += a[i * k + ki] * b[ki * n + j];
            }

            float actual = c[i * n + j];
            if (std::abs(actual - expected) > tolerance) {
                std::cerr << "Mismatch at (" << i << "," << j << "): expected "
                         << expected << ", got " << actual << std::endl;
                return false;
            }
        }
    }
    return true;
}

KPUSimulator::Config generate_multi_bank_config(size_t num_banks, size_t num_tiles) {
    KPUSimulator::Config config;
    config.memory_bank_count = num_banks;
    config.memory_bank_capacity_mb = 512; // Smaller banks for multi-bank setup
    config.memory_bandwidth_gbps = 16; // Higher bandwidth per bank
    config.scratchpad_count = num_tiles; // One scratchpad per tile
    config.scratchpad_capacity_kb = 256;
    config.compute_tile_count = num_tiles;
    config.dma_engine_count = num_banks + num_tiles; // Plenty of DMA engines
    return config;
}

bool run_distributed_matmul_test(KPUSimulator& sim, Size matrix_size) {
    // Generate test case
    auto test = generate_simple_matmul_test(matrix_size, matrix_size, matrix_size);

    // Use multiple banks and tiles if available
    size_t num_banks = sim.get_memory_bank_count();
    size_t num_tiles = sim.get_compute_tile_count();

    if (num_banks < 2 || num_tiles < 1) {
        std::cout << "Warning: Not enough banks/tiles for distributed test, using defaults" << std::endl;
        return sim.run_matmul_test(test);
    }

    std::cout << "Running distributed matmul test with " << num_banks
              << " banks and " << num_tiles << " tiles..." << std::endl;

    // For now, just use the first bank and tile (can be extended for true distribution)
    bool result = sim.run_matmul_test(test, 0, 0, 0);

    if (result) {
        std::cout << "Distributed test passed!" << std::endl;
        sim.print_component_status();
    }

    return result;
}

} // namespace test_utils

// KPUSimulator systolic array methods (must be in sw::kpu namespace)
bool KPUSimulator::is_using_systolic_arrays() const {
    return !compute_tiles.empty() &&
           compute_tiles[0].get_compute_type() == ComputeFabric::ComputeType::SYSTOLIC_ARRAY;
}

Size KPUSimulator::get_systolic_array_rows(size_t tile_id) const {
    if (tile_id >= compute_tiles.size()) {
        throw std::out_of_range("Invalid compute tile ID");
    }
    return compute_tiles[tile_id].get_systolic_rows();
}

Size KPUSimulator::get_systolic_array_cols(size_t tile_id) const {
    if (tile_id >= compute_tiles.size()) {
        throw std::out_of_range("Invalid compute tile ID");
    }
    return compute_tiles[tile_id].get_systolic_cols();
}

Size KPUSimulator::get_systolic_array_total_pes(size_t tile_id) const {
    if (tile_id >= compute_tiles.size()) {
        throw std::out_of_range("Invalid compute tile ID");
    }
    return get_systolic_array_rows(tile_id) * get_systolic_array_cols(tile_id);
}

// Tracing control methods
void KPUSimulator::enable_dma_tracing(size_t dma_id) {
    validate_dma_id(dma_id);
    dma_engines[dma_id].enable_tracing(true, &trace::TraceLogger::instance());
}

void KPUSimulator::enable_block_mover_tracing(size_t mover_id) {
    validate_block_mover_id(mover_id);
    block_movers[mover_id].enable_tracing();
}

void KPUSimulator::enable_streamer_tracing(size_t streamer_id) {
    validate_streamer_id(streamer_id);
    streamers[streamer_id].enable_tracing();
}

void KPUSimulator::enable_compute_fabric_tracing(size_t tile_id) {
    validate_tile_id(tile_id);
    compute_tiles[tile_id].enable_tracing();
}

void KPUSimulator::disable_dma_tracing(size_t dma_id) {
    validate_dma_id(dma_id);
    dma_engines[dma_id].enable_tracing(false);
}

void KPUSimulator::disable_block_mover_tracing(size_t mover_id) {
    validate_block_mover_id(mover_id);
    block_movers[mover_id].disable_tracing();
}

void KPUSimulator::disable_streamer_tracing(size_t streamer_id) {
    validate_streamer_id(streamer_id);
    streamers[streamer_id].disable_tracing();
}

void KPUSimulator::disable_compute_fabric_tracing(size_t tile_id) {
    validate_tile_id(tile_id);
    compute_tiles[tile_id].disable_tracing();
}

} // namespace sw::kpu