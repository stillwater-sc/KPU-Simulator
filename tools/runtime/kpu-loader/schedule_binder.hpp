/**
 * @file schedule_binder.hpp
 * @brief Binds abstract KIR operations to concrete hardware resources
 *
 * The schedule binder is the "driver" component that maps hardware-agnostic
 * KIR operations to specific micro-architecture resources.
 */

#pragma once

#include <sw/compiler/kir/kir.hpp>
#include <sw/kpu/kpu_simulator.hpp>
#include <vector>
#include <memory>

namespace sw::kpu::runtime {

using namespace sw::kpu::compiler;

/**
 * @brief Bound operation - KIR operation with concrete resource assignments
 */
struct BoundOperation {
    const kir::Operation* kir_op;   ///< Original KIR operation

    // Resource assignments
    size_t dma_engine_id;           ///< DMA engine assignment
    size_t block_mover_id;          ///< BlockMover assignment
    size_t streamer_id;             ///< Streamer assignment

    // Memory allocations
    uint64_t source_addr;           ///< Concrete source address
    uint64_t dest_addr;             ///< Concrete destination address
    size_t l2_bank_id;              ///< L2 bank assignment
    size_t l3_tile_id;              ///< L3 tile assignment
    size_t l1_buffer_id;            ///< L1 buffer assignment

    // Timing
    uint64_t start_cycle;           ///< Scheduled start cycle
    uint64_t end_cycle;             ///< Expected end cycle
};

/**
 * @brief Complete bound schedule
 */
struct BoundSchedule {
    const kir::Program* program;
    std::vector<BoundOperation> operations;

    // Resource utilization
    struct ResourceStats {
        size_t dma_engines_used;
        size_t block_movers_used;
        size_t streamers_used;
        size_t l2_banks_used;
        size_t l3_tiles_used;
        size_t l1_buffers_used;
    } resources;

    // Timing
    uint64_t total_cycles;
    double estimated_throughput;    ///< TFLOPS
};

/**
 * @brief Binds KIR programs to concrete hardware resources
 */
class ScheduleBinder {
public:
    /**
     * @brief Constructor
     * @param config KPU simulator configuration
     */
    explicit ScheduleBinder(const KPUSimulator::Config& config);

    /**
     * @brief Bind a KIR program to hardware resources
     *
     * @param program KIR program to bind
     * @return Bound schedule with concrete resource assignments
     */
    BoundSchedule bind(const kir::Program& program);

    /**
     * @brief Set simulator configuration
     */
    void set_config(const KPUSimulator::Config& config) { config_ = config; }

private:
    KPUSimulator::Config config_;

    /**
     * @brief Allocate L3 tile for a tensor tile
     */
    size_t allocate_l3_tile(const kir::TileSpec& tile);

    /**
     * @brief Allocate L2 bank for a tensor tile
     */
    size_t allocate_l2_bank(const kir::TileSpec& tile);

    /**
     * @brief Allocate L1 buffer for a tensor tile
     */
    size_t allocate_l1_buffer(const kir::TileSpec& tile);

    /**
     * @brief Assign DMA engine for a data movement operation
     */
    size_t assign_dma_engine(const kir::DataMoveOp& op);

    /**
     * @brief Assign BlockMover for L3↔L2 transfer
     */
    size_t assign_block_mover(const kir::DataMoveOp& op);

    /**
     * @brief Assign Streamer for L2↔L1 transfer
     */
    size_t assign_streamer(const kir::DataMoveOp& op);

    /**
     * @brief Calculate concrete memory address for a tile
     */
    uint64_t calculate_address(const kir::TileSpec& tile, kir::MemoryLevel level);
};

} // namespace sw::kpu::runtime
