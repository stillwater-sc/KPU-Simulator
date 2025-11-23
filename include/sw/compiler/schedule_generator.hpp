/**
 * @file schedule_generator.hpp
 * @brief Schedule generator for tiled matrix multiplication
 *
 * Converts TileOptimizer output into executable hardware command sequences
 * for DMA engines, BlockMovers, Streamers, and Compute fabric.
 */

#pragma once

#include <sw/compiler/tile_optimizer.hpp>
#include <sw/concepts.hpp>
#include <vector>
#include <memory>
#include <functional>
#include <cstdint>
#include <string>

namespace sw::kpu::compiler {

/**
 * @brief Schedule generator for matrix multiplication
 *
 * Takes TileConfig from TileOptimizer and generates executable schedules
 * for the KPU memory hierarchy and compute fabric.
 *
 * Features:
 * - Double-buffering for overlap of compute and data movement
 * - Pipelining across memory hierarchy levels
 * - Dependency tracking between commands
 * - Cycle-accurate timing estimation
 * - Memory address management
 */
class ScheduleGenerator {
public:
    /// Memory hierarchy levels
    enum class MemoryLevel {
        HOST_DDR,       // Host system memory
        KPU_GDDR6,      // KPU main memory banks
        L3_TILE,        // L3 cache tiles (128KB×4)
        L2_BANK,        // L2 cache banks (64KB×8)
        L1_BUFFER,      // L1 scratchpad buffers (32KB×4)
        PE_REGISTERS    // Systolic array PE registers
    };

    /// Command types in execution schedule
    enum class CommandType {
        DMA_TRANSFER,      // GDDR6 ↔ L3
        BLOCK_MOVE,        // L3 ↔ L2 with optional transform
        STREAM_L2_TO_L1,   // L2 → L1 streaming
        STREAM_L1_TO_L2,   // L1 → L2 streaming
        COMPUTE_MATMUL,    // Systolic array execution
        BARRIER            // Synchronization point
    };

    /// Single command in the schedule
    struct Command {
        CommandType type;

        // Source and destination
        MemoryLevel src_level;
        MemoryLevel dst_level;
        size_t src_id;          // Component ID (tile/bank/buffer)
        size_t dst_id;
        Address src_addr;
        Address dst_addr;

        // Data geometry
        Size height;            // Rows (for matrices/blocks)
        Size width;             // Columns
        Size size_bytes;        // Total transfer size

        // Compute-specific parameters
        Size M, N, K;           // For MATMUL commands

        // Timing
        Cycle issue_cycle;      // When command is issued
        Cycle start_cycle;      // When execution begins
        Cycle end_cycle;        // When execution completes
        Size latency_cycles;    // Duration

        // Dependencies
        std::vector<size_t> depends_on;  // Indices of prerequisite commands

        // Buffer allocation (for double-buffering)
        int buffer_id;          // Which buffer slot (0 or 1)

        // Metadata
        std::string tile_label; // e.g., "A[0,0]", "B[1,2]", "C[0,1]"
        uint64_t transaction_id;

        Command() : type(CommandType::BARRIER),
                   src_level(MemoryLevel::HOST_DDR),
                   dst_level(MemoryLevel::HOST_DDR),
                   src_id(0), dst_id(0), src_addr(0), dst_addr(0),
                   height(0), width(0), size_bytes(0),
                   M(0), N(0), K(0),
                   issue_cycle(0), start_cycle(0), end_cycle(0), latency_cycles(0),
                   buffer_id(0), transaction_id(0) {}
    };

    /// Complete schedule for matrix multiplication
    struct Schedule {
        // Original matrix dimensions
        Size M, N, K;

        // Tile configuration
        TileOptimizer::TileConfig config;

        // Command sequence
        std::vector<Command> commands;

        // Memory allocation map
        struct MemoryAllocation {
            MemoryLevel level;
            size_t component_id;
            Address base_addr;
            Size size_bytes;
            std::string label;
        };
        std::vector<MemoryAllocation> allocations;

        // Performance estimates
        Cycle total_cycles;
        Size total_dram_bytes;
        Size total_l3_bytes;
        Size total_l2_bytes;
        double estimated_time_ms;  // At 1 GHz
        double arithmetic_intensity;

        // Statistics
        size_t num_dma_transfers;
        size_t num_block_moves;
        size_t num_streams;
        size_t num_computes;
        size_t num_barriers;

        Schedule() : M(0), N(0), K(0),
                    total_cycles(0), total_dram_bytes(0),
                    total_l3_bytes(0), total_l2_bytes(0),
                    estimated_time_ms(0.0), arithmetic_intensity(0.0),
                    num_dma_transfers(0), num_block_moves(0),
                    num_streams(0), num_computes(0), num_barriers(0) {}
    };

    /// Memory hierarchy performance parameters
    struct PerformanceModel {
        // Bandwidth (GB/s)
        double dram_bandwidth;      // GDDR6 bandwidth
        double l3_bandwidth;        // L3 aggregate bandwidth
        double l2_bandwidth;        // L2 aggregate bandwidth
        double l1_bandwidth;        // L1 streaming bandwidth

        // Latency (cycles)
        Cycle dram_latency;         // DRAM access latency
        Cycle l3_latency;           // L3 access latency
        Cycle l2_latency;           // L2 access latency
        Cycle l1_latency;           // L1 access latency

        // Compute
        double clock_freq_ghz;      // System clock frequency
        Cycle systolic_latency;     // Systolic array startup latency

        // Default KPU configuration
        PerformanceModel() :
            dram_bandwidth(100.0),   // 100 GB/s GDDR6
            l3_bandwidth(400.0),     // 400 GB/s aggregate
            l2_bandwidth(800.0),     // 800 GB/s aggregate
            l1_bandwidth(1600.0),    // 1600 GB/s streaming
            dram_latency(100),
            l3_latency(20),
            l2_latency(10),
            l1_latency(5),
            clock_freq_ghz(1.0),
            systolic_latency(16) {}
    };

    /// Scheduling strategy
    enum class Strategy {
        SEQUENTIAL,         // Simple sequential execution (no overlap)
        DOUBLE_BUFFERED,    // Double-buffer each level
        FULLY_PIPELINED     // Full pipeline with all stages overlapped
    };

public:
    explicit ScheduleGenerator(const PerformanceModel& perf = PerformanceModel());
    ~ScheduleGenerator() = default;

    /**
     * @brief Generate execution schedule for matrix multiplication
     *
     * @param M Rows of A and C
     * @param N Columns of B and C
     * @param K Columns of A, rows of B
     * @param config Tile configuration from TileOptimizer
     * @param strategy Scheduling strategy
     * @return Complete execution schedule
     */
    Schedule generate(Size M, Size N, Size K,
                     const TileOptimizer::TileConfig& config,
                     Strategy strategy = Strategy::DOUBLE_BUFFERED);

    /**
     * @brief Allocate memory addresses for matrices
     *
     * Assigns addresses in each memory level for A, B, C matrices
     * and intermediate tiles.
     */
    void allocate_memory(Schedule& schedule);

    /**
     * @brief Generate DMA commands for DRAM ↔ L3 transfers
     *
     * Creates commands to load A and B tiles from DRAM to L3,
     * and store C tiles from L3 to DRAM.
     */
    void generate_dma_commands(Schedule& schedule, Size M, Size N, Size K);

    /**
     * @brief Generate BlockMover commands for L3 ↔ L2 transfers
     */
    void generate_block_move_commands(Schedule& schedule, Size M, Size N, Size K);

    /**
     * @brief Generate Streamer commands for L2 ↔ L1 streaming
     */
    void generate_stream_commands(Schedule& schedule, Size M, Size N, Size K);

    /**
     * @brief Generate Compute commands for systolic array
     */
    void generate_compute_commands(Schedule& schedule, Size M, Size N, Size K);

    /**
     * @brief Calculate dependencies between commands
     *
     * Determines execution order constraints based on data dependencies.
     */
    void calculate_dependencies(Schedule& schedule);

    /**
     * @brief Estimate timing for all commands
     *
     * Computes issue/start/end cycles based on dependencies and latencies.
     */
    void estimate_timing(Schedule& schedule);

    /**
     * @brief Apply double-buffering optimization
     *
     * Overlaps compute with data movement by using two buffer sets.
     */
    void apply_double_buffering(Schedule& schedule);

    /**
     * @brief Apply full pipelining optimization
     *
     * Maximally overlaps all pipeline stages across multiple tiles.
     */
    void apply_pipelining(Schedule& schedule);

    /**
     * @brief Validate schedule for correctness
     *
     * Checks memory constraints, dependency ordering, address validity.
     */
    bool validate(const Schedule& schedule, std::string& error_msg) const;

    /**
     * @brief Print schedule in human-readable format
     */
    void print_schedule(const Schedule& schedule, bool verbose = false) const;

    /**
     * @brief Export schedule to JSON format
     */
    std::string export_json(const Schedule& schedule) const;

    // Accessors
    const PerformanceModel& performance_model() const { return perf_; }
    void set_performance_model(const PerformanceModel& perf) { perf_ = perf; }

private:
    PerformanceModel perf_;

    // Helper methods
    Cycle calculate_transfer_cycles(Size bytes, double bandwidth_gb_s) const;
    Cycle calculate_compute_cycles(Size M, Size N, Size K) const;

    void add_dma_command(Schedule& schedule, const std::string& label,
                        Address src_addr, Address dst_addr, Size bytes);

    void add_block_move_command(Schedule& schedule, const std::string& label,
                               size_t src_tile, Address src_addr,
                               size_t dst_bank, Address dst_addr,
                               Size height, Size width);

    void add_stream_command(Schedule& schedule, const std::string& label,
                           bool is_l2_to_l1,
                           size_t bank_id, size_t buffer_id,
                           Address l2_addr, Address l1_addr,
                           Size height, Size width);

    void add_compute_command(Schedule& schedule, const std::string& label,
                            size_t buffer_id,
                            Address a_addr, Address b_addr, Address c_addr,
                            Size M, Size N, Size K);

    void add_barrier(Schedule& schedule, const std::string& label);

    // Memory address management
    struct AddressAllocator {
        Address current_offset;
        Size capacity;

        Address allocate(Size bytes) {
            Address addr = current_offset;
            current_offset += bytes;
            return addr;
        }

        void reset() { current_offset = 0; }
    };

    // Tile indexing helpers
    struct TileIndex {
        Size ti, tj, tk;  // Tile coordinates in M, N, K dimensions

        // Create label for A tile: A_tile[ti, tk]
        std::string label_A() const {
            return "A_tile[" + std::to_string(ti) + "," + std::to_string(tk) + "]";
        }

        // Create label for B tile: B_tile[tk, tj]
        std::string label_B() const {
            return "B_tile[" + std::to_string(tk) + "," + std::to_string(tj) + "]";
        }

        // Create label for C tile: C_tile[ti, tj]
        std::string label_C() const {
            return "C_tile[" + std::to_string(ti) + "," + std::to_string(tj) + "]";
        }

        // Legacy label method for backwards compatibility
        std::string label(char matrix) const {
            if (matrix == 'A') return label_A();
            if (matrix == 'B') return label_B();
            return label_C();
        }
    };

    Size tile_count(Size dim, Size tile_size) const {
        return (dim + tile_size - 1) / tile_size;
    }

    Size tile_dimension(Size global_dim, Size tile_idx, Size tile_size) const {
        return std::min(tile_size, global_dim - tile_idx * tile_size);
    }
};

} // namespace sw::kpu::compiler
