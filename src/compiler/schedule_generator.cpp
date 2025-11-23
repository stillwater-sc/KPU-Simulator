/**
 * @file schedule_generator.cpp
 * @brief Implementation of schedule generator for tiled matrix multiplication
 */

#include <sw/compiler/schedule_generator.hpp>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace sw::kpu::compiler {

ScheduleGenerator::ScheduleGenerator(const PerformanceModel& perf)
    : perf_(perf) {}

ScheduleGenerator::Schedule ScheduleGenerator::generate(Size M, Size N, Size K,
                                     const TileOptimizer::TileConfig& config,
                                     Strategy strategy) {
    Schedule schedule;
    schedule.M = M;
    schedule.N = N;
    schedule.K = K;
    schedule.config = config;

    // Allocate memory addresses
    allocate_memory(schedule);

    // Generate base commands
    generate_dma_commands(schedule, M, N, K);
    generate_block_move_commands(schedule, M, N, K);
    generate_stream_commands(schedule, M, N, K);
    generate_compute_commands(schedule, M, N, K);

    // Calculate dependencies
    calculate_dependencies(schedule);

    // Apply optimization strategy
    switch (strategy) {
        case Strategy::SEQUENTIAL:
            // No optimization - commands execute in order
            break;

        case Strategy::DOUBLE_BUFFERED:
            apply_double_buffering(schedule);
            break;

        case Strategy::FULLY_PIPELINED:
            apply_pipelining(schedule);
            break;
    }

    // Estimate timing
    estimate_timing(schedule);

    // Calculate statistics
    schedule.total_cycles = 0;
    for (const auto& cmd : schedule.commands) {
        schedule.total_cycles = std::max(schedule.total_cycles, cmd.end_cycle);

        switch (cmd.type) {
            case CommandType::DMA_TRANSFER:
                schedule.num_dma_transfers++;
                schedule.total_dram_bytes += cmd.size_bytes;
                break;
            case CommandType::BLOCK_MOVE:
                schedule.num_block_moves++;
                schedule.total_l3_bytes += cmd.size_bytes;
                break;
            case CommandType::STREAM_L2_TO_L1:
            case CommandType::STREAM_L1_TO_L2:
                schedule.num_streams++;
                schedule.total_l2_bytes += cmd.size_bytes;
                break;
            case CommandType::COMPUTE_MATMUL:
                schedule.num_computes++;
                break;
            case CommandType::BARRIER:
                schedule.num_barriers++;
                break;
        }
    }

    schedule.estimated_time_ms = schedule.total_cycles / (perf_.clock_freq_ghz * 1e6);

    Size total_flops = 2 * M * N * K;
    schedule.arithmetic_intensity = static_cast<double>(total_flops) / schedule.total_dram_bytes;

    return schedule;
}

void ScheduleGenerator::allocate_memory(Schedule& schedule) {
    const Size elem_size = 4;  // float32
    const Size Ti = schedule.config.Ti;
    const Size Tj = schedule.config.Tj;
    const Size Tk = schedule.config.Tk;

    // GDDR6 allocations (full matrices)
    AddressAllocator gddr6_alloc{0, 512 * 1024 * 1024};  // 512 MB

    Address gddr6_a_addr = gddr6_alloc.allocate(schedule.M * schedule.K * elem_size);
    Address gddr6_b_addr = gddr6_alloc.allocate(schedule.K * schedule.N * elem_size);
    Address gddr6_c_addr = gddr6_alloc.allocate(schedule.M * schedule.N * elem_size);

    schedule.allocations.push_back({MemoryLevel::KPU_GDDR6, 0, gddr6_a_addr,
                                    schedule.M * schedule.K * elem_size, "Matrix A"});
    schedule.allocations.push_back({MemoryLevel::KPU_GDDR6, 0, gddr6_b_addr,
                                    schedule.K * schedule.N * elem_size, "Matrix B"});
    schedule.allocations.push_back({MemoryLevel::KPU_GDDR6, 0, gddr6_c_addr,
                                    schedule.M * schedule.N * elem_size, "Matrix C"});

    // L3 allocations (tiles - double buffered)
    AddressAllocator l3_alloc{0, 128 * 1024};  // 128 KB per tile

    Size a_tile_size = Ti * Tk * elem_size;
    Size b_tile_size = Tk * Tj * elem_size;
    Size c_tile_size = Ti * Tj * elem_size;

    Address l3_a_addr_0 = l3_alloc.allocate(a_tile_size);
    Address l3_a_addr_1 = l3_alloc.allocate(a_tile_size);
    Address l3_b_addr_0 = l3_alloc.allocate(b_tile_size);
    Address l3_b_addr_1 = l3_alloc.allocate(b_tile_size);
    Address l3_c_addr = l3_alloc.allocate(c_tile_size);

    schedule.allocations.push_back({MemoryLevel::L3_TILE, 0, l3_a_addr_0, a_tile_size, "A tile buf0"});
    schedule.allocations.push_back({MemoryLevel::L3_TILE, 0, l3_a_addr_1, a_tile_size, "A tile buf1"});
    schedule.allocations.push_back({MemoryLevel::L3_TILE, 0, l3_b_addr_0, b_tile_size, "B tile buf0"});
    schedule.allocations.push_back({MemoryLevel::L3_TILE, 0, l3_b_addr_1, b_tile_size, "B tile buf1"});
    schedule.allocations.push_back({MemoryLevel::L3_TILE, 0, l3_c_addr, c_tile_size, "C tile"});

    // L2 allocations (double buffered)
    AddressAllocator l2_alloc{0, 64 * 1024};  // 64 KB per bank

    Address l2_a_addr_0 = l2_alloc.allocate(a_tile_size);
    Address l2_a_addr_1 = l2_alloc.allocate(a_tile_size);
    Address l2_b_addr_0 = l2_alloc.allocate(b_tile_size);
    Address l2_b_addr_1 = l2_alloc.allocate(b_tile_size);
    Address l2_c_addr = l2_alloc.allocate(c_tile_size);

    schedule.allocations.push_back({MemoryLevel::L2_BANK, 0, l2_a_addr_0, a_tile_size, "A L2 buf0"});
    schedule.allocations.push_back({MemoryLevel::L2_BANK, 0, l2_a_addr_1, a_tile_size, "A L2 buf1"});
    schedule.allocations.push_back({MemoryLevel::L2_BANK, 0, l2_b_addr_0, b_tile_size, "B L2 buf0"});
    schedule.allocations.push_back({MemoryLevel::L2_BANK, 0, l2_b_addr_1, b_tile_size, "B L2 buf1"});
    schedule.allocations.push_back({MemoryLevel::L2_BANK, 0, l2_c_addr, c_tile_size, "C L2"});

    // L1 allocations
    AddressAllocator l1_alloc{0, 32 * 1024};  // 32 KB per buffer

    Address l1_a_addr = l1_alloc.allocate(a_tile_size);
    Address l1_b_addr = l1_alloc.allocate(b_tile_size);
    Address l1_c_addr = l1_alloc.allocate(c_tile_size);

    schedule.allocations.push_back({MemoryLevel::L1_BUFFER, 0, l1_a_addr, a_tile_size, "A L1"});
    schedule.allocations.push_back({MemoryLevel::L1_BUFFER, 0, l1_b_addr, b_tile_size, "B L1"});
    schedule.allocations.push_back({MemoryLevel::L1_BUFFER, 0, l1_c_addr, c_tile_size, "C L1"});
}

void ScheduleGenerator::generate_dma_commands(Schedule& schedule, Size M, Size N, Size K) {
    const Size elem_size = 4;
    // const Size Ti = schedule.config.Ti;
    // const Size Tj = schedule.config.Tj;
    // const Size Tk = schedule.config.Tk;

    // Size m_tiles = tile_count(M, Ti);
    // Size n_tiles = tile_count(N, Tj);
    // Size k_tiles = tile_count(K, Tk);

    // For now, generate simple load/store commands
    // Full implementation would generate per-tile DMA commands

    // Load A matrix from GDDR6
    Size a_bytes = M * K * elem_size;
    add_dma_command(schedule, "Load A", 0x0000, 0x0000, a_bytes);

    // Load B matrix from GDDR6
    Size b_bytes = K * N * elem_size;
    add_dma_command(schedule, "Load B", 0x0000, 0x0000, b_bytes);

    // Store C matrix to GDDR6 (at the end)
    Size c_bytes = M * N * elem_size;
    add_dma_command(schedule, "Store C", 0x0000, 0x0000, c_bytes);
}

void ScheduleGenerator::generate_block_move_commands(Schedule& schedule, Size M, Size N, Size K) {
    const Size Ti = schedule.config.Ti;
    const Size Tj = schedule.config.Tj;
    const Size Tk = schedule.config.Tk;

    Size m_tiles = tile_count(M, Ti);
    Size n_tiles = tile_count(N, Tj);
    Size k_tiles = tile_count(K, Tk);

    // Generate L3→L2 block moves for each tile computation
    for (Size ti = 0; ti < m_tiles; ++ti) {
        for (Size tj = 0; tj < n_tiles; ++tj) {
            for (Size tk = 0; tk < k_tiles; ++tk) {
                TileIndex a_idx{ti, 0, tk};
                TileIndex b_idx{0, tj, tk};

                Size tile_m = tile_dimension(M, ti, Ti);
                Size tile_n = tile_dimension(N, tj, Tj);
                Size tile_k = tile_dimension(K, tk, Tk);

                // A tile: L3→L2
                add_block_move_command(schedule, a_idx.label('A'),
                                      0, 0x0000,  // L3 tile 0
                                      0, 0x0000,  // L2 bank 0
                                      tile_m, tile_k);

                // B tile: L3→L2
                add_block_move_command(schedule, b_idx.label('B'),
                                      0, 0x0000,
                                      0, 0x0000,
                                      tile_k, tile_n);
            }
        }
    }
}

void ScheduleGenerator::generate_stream_commands(Schedule& schedule, Size M, Size N, Size K) {
    const Size Ti = schedule.config.Ti;
    const Size Tj = schedule.config.Tj;
    const Size Tk = schedule.config.Tk;

    Size m_tiles = tile_count(M, Ti);
    Size n_tiles = tile_count(N, Tj);
    Size k_tiles = tile_count(K, Tk);

    // Generate L2→L1 streams for each tile computation
    for (Size ti = 0; ti < m_tiles; ++ti) {
        for (Size tj = 0; tj < n_tiles; ++tj) {
            for (Size tk = 0; tk < k_tiles; ++tk) {
                TileIndex a_idx{ti, 0, tk};
                TileIndex b_idx{0, tj, tk};

                Size tile_m = tile_dimension(M, ti, Ti);
                Size tile_n = tile_dimension(N, tj, Tj);
                Size tile_k = tile_dimension(K, tk, Tk);

                // Stream A: L2→L1
                add_stream_command(schedule, a_idx.label('A') + " L2→L1",
                                  true, 0, 0, 0x0000, 0x0000,
                                  tile_m, tile_k);

                // Stream B: L2→L1
                add_stream_command(schedule, b_idx.label('B') + " L2→L1",
                                  true, 0, 0, 0x0000, 0x0000,
                                  tile_k, tile_n);
            }
        }
    }

    // Generate L1→L2 streams for results
    for (Size ti = 0; ti < m_tiles; ++ti) {
        for (Size tj = 0; tj < n_tiles; ++tj) {
            TileIndex c_idx{ti, tj, 0};
            Size tile_m = tile_dimension(M, ti, Ti);
            Size tile_n = tile_dimension(N, tj, Tj);

            add_stream_command(schedule, c_idx.label('C') + " L1→L2",
                              false, 0, 0, 0x0000, 0x0000,
                              tile_m, tile_n);
        }
    }
}

void ScheduleGenerator::generate_compute_commands(Schedule& schedule, Size M, Size N, Size K) {
    const Size Ti = schedule.config.Ti;
    const Size Tj = schedule.config.Tj;
    const Size Tk = schedule.config.Tk;

    Size m_tiles = tile_count(M, Ti);
    Size n_tiles = tile_count(N, Tj);
    Size k_tiles = tile_count(K, Tk);

    // Generate compute commands for each C tile
    for (Size ti = 0; ti < m_tiles; ++ti) {
        for (Size tj = 0; tj < n_tiles; ++tj) {
            for (Size tk = 0; tk < k_tiles; ++tk) {
                TileIndex c_idx{ti, tj, tk};

                Size tile_m = tile_dimension(M, ti, Ti);
                Size tile_n = tile_dimension(N, tj, Tj);
                Size tile_k = tile_dimension(K, tk, Tk);

                add_compute_command(schedule, c_idx.label('C'),
                                   0, 0x0000, 0x0000, 0x0000,
                                   tile_m, tile_n, tile_k);
            }
        }
    }
}

void ScheduleGenerator::calculate_dependencies(Schedule& schedule) {
    // Simple dependency model: commands execute in order
    // Each command depends on all previous commands

    for (size_t i = 1; i < schedule.commands.size(); ++i) {
        // For now, simple sequential dependency
        schedule.commands[i].depends_on.push_back(i - 1);
    }

    // TODO: Implement smarter dependency analysis based on:
    // - Read-after-write (RAW) dependencies
    // - Write-after-read (WAR) dependencies
    // - Write-after-write (WAW) dependencies
    // - Memory level (L3→L2→L1→Compute→L1→L2→L3)
}

void ScheduleGenerator::estimate_timing(Schedule& schedule) {
    // Initialize all commands to cycle 0
    for (auto& cmd : schedule.commands) {
        cmd.issue_cycle = 0;
        cmd.start_cycle = 0;
        cmd.end_cycle = 0;
    }

    // Calculate latencies
    for (auto& cmd : schedule.commands) {
        switch (cmd.type) {
            case CommandType::DMA_TRANSFER:
                cmd.latency_cycles = calculate_transfer_cycles(cmd.size_bytes, perf_.dram_bandwidth) +
                                    perf_.dram_latency;
                break;

            case CommandType::BLOCK_MOVE:
                cmd.latency_cycles = calculate_transfer_cycles(cmd.size_bytes, perf_.l3_bandwidth) +
                                    perf_.l3_latency;
                break;

            case CommandType::STREAM_L2_TO_L1:
            case CommandType::STREAM_L1_TO_L2:
                cmd.latency_cycles = calculate_transfer_cycles(cmd.size_bytes, perf_.l2_bandwidth) +
                                    perf_.l2_latency;
                break;

            case CommandType::COMPUTE_MATMUL:
                cmd.latency_cycles = calculate_compute_cycles(cmd.M, cmd.N, cmd.K) +
                                    perf_.systolic_latency;
                break;

            case CommandType::BARRIER:
                cmd.latency_cycles = 1;
                break;
        }
    }

    // Schedule commands based on dependencies
    Cycle current_cycle = 0;

    for (size_t i = 0; i < schedule.commands.size(); ++i) {
        auto& cmd = schedule.commands[i];

        // Start after all dependencies complete
        Cycle earliest_start = 0;
        for (size_t dep_idx : cmd.depends_on) {
            earliest_start = std::max(earliest_start, schedule.commands[dep_idx].end_cycle);
        }

        cmd.issue_cycle = earliest_start;
        cmd.start_cycle = earliest_start;
        cmd.end_cycle = cmd.start_cycle + cmd.latency_cycles;

        current_cycle = std::max(current_cycle, cmd.end_cycle);
    }
}

void ScheduleGenerator::apply_double_buffering([[maybe_unused]] Schedule& schedule) {
    // TODO: Implement double-buffering optimization
    // - Allocate two buffer sets
    // - Overlap compute on buffer[0] with load to buffer[1]
    // - Swap buffers after each tile
}

void ScheduleGenerator::apply_pipelining([[maybe_unused]] Schedule& schedule) {
    // TODO: Implement full pipeline optimization
    // - Create pipeline stages: DMA, BlockMove, Stream, Compute
    // - Allow multiple tiles in flight
    // - Maximize stage overlap
}

bool ScheduleGenerator::validate(const Schedule& schedule, std::string& error_msg) const {
    // Check that tile sizes fit in L2
    Size l2_footprint = schedule.config.l2_footprint;
    if (l2_footprint > 64 * 1024) {
        error_msg = "L2 footprint exceeds 64KB";
        return false;
    }

    // Check that all commands have valid cycles
    for (const auto& cmd : schedule.commands) {
        if (cmd.end_cycle < cmd.start_cycle) {
            error_msg = "Invalid timing: end_cycle < start_cycle";
            return false;
        }
    }

    // Check dependencies
    for (size_t i = 0; i < schedule.commands.size(); ++i) {
        for (size_t dep : schedule.commands[i].depends_on) {
            if (dep >= i) {
                error_msg = "Invalid dependency: forward reference";
                return false;
            }
        }
    }

    error_msg = "Valid";
    return true;
}

void ScheduleGenerator::print_schedule(const Schedule& schedule, bool verbose) const {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Schedule for C[" << schedule.M << "," << schedule.N << "] = "
              << "A[" << schedule.M << "," << schedule.K << "] × "
              << "B[" << schedule.K << "," << schedule.N << "]\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << "\nTile Configuration:\n";
    std::cout << "  Ti × Tj × Tk: " << schedule.config.Ti << " × "
              << schedule.config.Tj << " × " << schedule.config.Tk << "\n";
    std::cout << "  Tile count: "
              << tile_count(schedule.M, schedule.config.Ti) << " × "
              << tile_count(schedule.N, schedule.config.Tj) << " × "
              << tile_count(schedule.K, schedule.config.Tk) << "\n";

    std::cout << "\nMemory Allocations:\n";
    for (const auto& alloc : schedule.allocations) {
        std::cout << "  " << std::setw(12) << std::left << alloc.label
                  << ": 0x" << std::hex << std::setw(8) << std::setfill('0')
                  << alloc.base_addr << std::dec << std::setfill(' ')
                  << " (" << (alloc.size_bytes / 1024.0) << " KB)\n";
    }

    std::cout << "\nCommand Statistics:\n";
    std::cout << "  Total commands:  " << schedule.commands.size() << "\n";
    std::cout << "  DMA transfers:   " << schedule.num_dma_transfers << "\n";
    std::cout << "  Block moves:     " << schedule.num_block_moves << "\n";
    std::cout << "  Streams:         " << schedule.num_streams << "\n";
    std::cout << "  Computes:        " << schedule.num_computes << "\n";
    std::cout << "  Barriers:        " << schedule.num_barriers << "\n";

    std::cout << "\nPerformance Estimates:\n";
    std::cout << "  Total cycles:    " << schedule.total_cycles << "\n";
    std::cout << "  Estimated time:  " << std::fixed << std::setprecision(3)
              << schedule.estimated_time_ms << " ms\n";
    std::cout << "  DRAM traffic:    " << (schedule.total_dram_bytes / (1024.0 * 1024.0))
              << " MB\n";
    std::cout << "  Arithmetic int:  " << std::setprecision(2)
              << schedule.arithmetic_intensity << " FLOPs/byte\n";

    if (verbose) {
        std::cout << "\nCommand Timeline:\n";
        std::cout << std::string(70, '-') << "\n";

        const char* type_names[] = {
            "DMA", "BlockMove", "Stream→L1", "Stream→L2", "Compute", "Barrier"
        };

        for (size_t i = 0; i < schedule.commands.size(); ++i) {
            const auto& cmd = schedule.commands[i];
            std::cout << std::setw(4) << i << " | "
                      << std::setw(10) << std::left << type_names[static_cast<int>(cmd.type)]
                      << " | " << std::setw(20) << cmd.tile_label
                      << " | " << std::setw(6) << cmd.start_cycle
                      << " → " << std::setw(6) << cmd.end_cycle
                      << " (" << std::setw(5) << cmd.latency_cycles << " cyc)\n";
        }
    }

    std::cout << std::string(70, '=') << "\n";
}

std::string ScheduleGenerator::export_json([[maybe_unused]] const Schedule& schedule) const {
    // TODO: Implement JSON export
    return "{}";
}

// Helper methods

Cycle ScheduleGenerator::calculate_transfer_cycles(Size bytes, double bandwidth_gb_s) const {
    // Convert bandwidth from GB/s to bytes/cycle
    double bytes_per_cycle = (bandwidth_gb_s * 1e9) / (perf_.clock_freq_ghz * 1e9);
    return static_cast<Cycle>(std::ceil(bytes / bytes_per_cycle));
}

Cycle ScheduleGenerator::calculate_compute_cycles(Size M, Size N, Size K) const {
    // For 16×16 systolic array: max(M, N, K) cycles for streaming
    // Plus accumulation time
    return std::max({M, N, K});
}

void ScheduleGenerator::add_dma_command(Schedule& schedule, const std::string& label,
                                       Address src_addr, Address dst_addr, Size bytes) {
    Command cmd;
    cmd.type = CommandType::DMA_TRANSFER;
    cmd.src_level = MemoryLevel::KPU_GDDR6;
    cmd.dst_level = MemoryLevel::L3_TILE;
    cmd.src_addr = src_addr;
    cmd.dst_addr = dst_addr;
    cmd.size_bytes = bytes;
    cmd.tile_label = label;
    cmd.buffer_id = 0;
    schedule.commands.push_back(cmd);
}

void ScheduleGenerator::add_block_move_command(Schedule& schedule, const std::string& label,
                                               size_t src_tile, Address src_addr,
                                               size_t dst_bank, Address dst_addr,
                                               Size height, Size width) {
    Command cmd;
    cmd.type = CommandType::BLOCK_MOVE;
    cmd.src_level = MemoryLevel::L3_TILE;
    cmd.dst_level = MemoryLevel::L2_BANK;
    cmd.src_id = src_tile;
    cmd.dst_id = dst_bank;
    cmd.src_addr = src_addr;
    cmd.dst_addr = dst_addr;
    cmd.height = height;
    cmd.width = width;
    cmd.size_bytes = height * width * 4;  // float32
    cmd.tile_label = label;
    cmd.buffer_id = 0;
    schedule.commands.push_back(cmd);
}

void ScheduleGenerator::add_stream_command(Schedule& schedule, const std::string& label,
                                           bool is_l2_to_l1,
                                           size_t bank_id, size_t buffer_id,
                                           Address l2_addr, Address l1_addr,
                                           Size height, Size width) {
    Command cmd;
    cmd.type = is_l2_to_l1 ? CommandType::STREAM_L2_TO_L1 : CommandType::STREAM_L1_TO_L2;
    cmd.src_level = is_l2_to_l1 ? MemoryLevel::L2_BANK : MemoryLevel::L1_BUFFER;
    cmd.dst_level = is_l2_to_l1 ? MemoryLevel::L1_BUFFER : MemoryLevel::L2_BANK;
    cmd.src_id = bank_id;
    cmd.dst_id = buffer_id;
    cmd.src_addr = is_l2_to_l1 ? l2_addr : l1_addr;
    cmd.dst_addr = is_l2_to_l1 ? l1_addr : l2_addr;
    cmd.height = height;
    cmd.width = width;
    cmd.size_bytes = height * width * 4;
    cmd.tile_label = label;
    cmd.buffer_id = 0;
    schedule.commands.push_back(cmd);
}

void ScheduleGenerator::add_compute_command(Schedule& schedule, const std::string& label,
                                            size_t buffer_id,
                                            Address a_addr, [[maybe_unused]] Address b_addr, Address c_addr,
                                            Size M, Size N, Size K) {
    Command cmd;
    cmd.type = CommandType::COMPUTE_MATMUL;
    cmd.src_level = MemoryLevel::L1_BUFFER;
    cmd.dst_level = MemoryLevel::L1_BUFFER;
    cmd.src_id = buffer_id;
    cmd.dst_id = buffer_id;
    cmd.src_addr = a_addr;
    cmd.dst_addr = c_addr;
    cmd.M = M;
    cmd.N = N;
    cmd.K = K;
    cmd.size_bytes = 0;  // Compute doesn't transfer data
    cmd.tile_label = label;
    cmd.buffer_id = 0;
    schedule.commands.push_back(cmd);
}

void ScheduleGenerator::add_barrier(Schedule& schedule, const std::string& label) {
    Command cmd;
    cmd.type = CommandType::BARRIER;
    cmd.tile_label = label;
    schedule.commands.push_back(cmd);
}

} // namespace sw::kpu::compiler
