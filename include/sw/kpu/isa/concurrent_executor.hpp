/**
 * @file concurrent_executor.hpp
 * @brief Concurrent execution model for Data Movement ISA
 *
 * The KPU has multiple hardware resources that execute concurrently:
 * - Multiple DMA engines (one per memory channel)
 * - Multiple BlockMovers (L3→L2)
 * - Multiple Streamers (L2→L1)
 * - Compute fabric (systolic array)
 *
 * This executor models the true concurrent nature of the architecture,
 * scheduling operations onto resources and tracking occupancy over time.
 */

#pragma once

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/concepts.hpp>
#include <vector>
#include <deque>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <functional>

namespace sw::kpu::isa {

// ============================================================================
// Resource Types
// ============================================================================

/**
 * @brief Resource type enumeration
 */
enum class ResourceType {
    DMA_ENGINE,
    BLOCK_MOVER,
    STREAMER,
    COMPUTE_FABRIC
};

/**
 * @brief Resource identifier (type + index)
 */
struct ResourceId {
    ResourceType type;
    uint8_t index;

    bool operator==(const ResourceId& other) const {
        return type == other.type && index == other.index;
    }

    bool operator<(const ResourceId& other) const {
        if (type != other.type) return type < other.type;
        return index < other.index;
    }

    std::string to_string() const {
        std::string type_str;
        switch (type) {
            case ResourceType::DMA_ENGINE: type_str = "DMA"; break;
            case ResourceType::BLOCK_MOVER: type_str = "BM"; break;
            case ResourceType::STREAMER: type_str = "STR"; break;
            case ResourceType::COMPUTE_FABRIC: type_str = "COMP"; break;
        }
        return type_str + "[" + std::to_string(index) + "]";
    }
};

// ============================================================================
// Operation in Flight
// ============================================================================

/**
 * @brief Represents an operation scheduled on a resource
 */
struct ScheduledOp {
    uint32_t instruction_id;
    ResourceId resource;
    Cycle start_cycle;
    Cycle end_cycle;
    std::string label;
    MatrixID matrix;
    TileCoord tile;

    Cycle duration() const { return end_cycle - start_cycle; }
};

// ============================================================================
// Hardware Resource Model
// ============================================================================

/**
 * @brief Models a hardware resource with timing and occupancy
 */
class HardwareResource {
public:
    ResourceId id;
    double bandwidth_gb_s;      // Bandwidth for this resource
    Cycle next_available_cycle; // When resource becomes free

    // Occupancy tracking
    std::vector<ScheduledOp> completed_ops;
    ScheduledOp* current_op;

    HardwareResource(ResourceType type, uint8_t index, double bandwidth = 100.0)
        : id{type, index}, bandwidth_gb_s(bandwidth),
          next_available_cycle(0), current_op(nullptr) {}

    bool is_busy(Cycle at_cycle) const {
        return at_cycle < next_available_cycle;
    }

    Cycle schedule_op(Cycle earliest, Size bytes, uint32_t instr_id,
                      const std::string& label, MatrixID mat, TileCoord tile) {
        Cycle start = std::max(earliest, next_available_cycle);
        // Calculate cycles based on bandwidth (assume 1 GHz clock)
        Cycle cycles = static_cast<Cycle>(bytes / bandwidth_gb_s);
        if (cycles == 0) cycles = 1;  // Minimum 1 cycle

        ScheduledOp op;
        op.instruction_id = instr_id;
        op.resource = id;
        op.start_cycle = start;
        op.end_cycle = start + cycles;
        op.label = label;
        op.matrix = mat;
        op.tile = tile;

        completed_ops.push_back(op);
        next_available_cycle = op.end_cycle;

        return op.end_cycle;
    }
};

// ============================================================================
// Memory Channel Model
// ============================================================================

/**
 * @brief Models a memory channel with its associated DMA engine
 *
 * Each memory channel has:
 * - One DMA engine for external memory transfers
 * - Associated bandwidth constraints
 * - Queue of pending transfers
 */
struct MemoryChannel {
    uint8_t channel_id;
    double bandwidth_gb_s;
    HardwareResource dma_engine;

    MemoryChannel(uint8_t id, double bw = 50.0)
        : channel_id(id), bandwidth_gb_s(bw),
          dma_engine(ResourceType::DMA_ENGINE, id, bw) {}
};

// ============================================================================
// System Resource Configuration
// ============================================================================

/**
 * @brief Configuration for system resources
 */
struct ResourceConfig {
    uint8_t num_memory_channels = 4;    // DMA engines
    uint8_t num_block_movers = 4;       // L3→L2 movers
    uint8_t num_streamers = 4;          // L2→L1 streamers

    double dma_bandwidth_gb_s = 50.0;       // Per channel
    double block_mover_bandwidth_gb_s = 100.0;
    double streamer_bandwidth_gb_s = 200.0;

    Size systolic_size = 16;            // 16x16 systolic array
    double compute_throughput_gflops = 1000.0;
};

// ============================================================================
// Concurrent Executor
// ============================================================================

/**
 * @brief Executes Data Movement programs with true concurrency model
 *
 * This executor:
 * 1. Schedules operations onto available resources
 * 2. Respects data dependencies
 * 3. Tracks resource occupancy over time
 * 4. Generates timeline visualizations
 */
class ConcurrentExecutor {
public:
    explicit ConcurrentExecutor(const ResourceConfig& config);

    /**
     * @brief Execute a program and collect timing information
     * @param program The Data Movement program to execute
     * @return Total execution cycles
     */
    Cycle execute(const DMProgram& program);

    /**
     * @brief Get resource utilization statistics
     */
    struct UtilizationStats {
        double dma_utilization;
        double block_mover_utilization;
        double streamer_utilization;
        double compute_utilization;
        Cycle total_cycles;
        Cycle makespan;  // Wall-clock cycles from start to finish
    };
    UtilizationStats get_utilization() const;

    /**
     * @brief Generate ASCII timeline visualization
     * @param width Character width of timeline
     * @return Multi-line string showing resource occupancy
     */
    std::string generate_timeline(size_t width = 80) const;

    /**
     * @brief Generate detailed cycle-by-cycle report
     */
    std::string generate_cycle_report() const;

    /**
     * @brief Get all scheduled operations for analysis
     */
    const std::vector<ScheduledOp>& get_all_operations() const { return all_ops_; }

private:
    ResourceConfig config_;

    // Hardware resources
    std::vector<MemoryChannel> memory_channels_;
    std::vector<HardwareResource> block_movers_;
    std::vector<HardwareResource> streamers_;
    HardwareResource compute_fabric_;

    // Execution state
    std::vector<ScheduledOp> all_ops_;
    std::map<uint32_t, Cycle> instruction_completion_; // instr_id -> completion cycle
    Cycle current_cycle_;
    Cycle makespan_;

    // Barrier tracking
    Cycle last_barrier_cycle_;

    // Schedule an instruction onto appropriate resource
    void schedule_instruction(const DMInstruction& instr);

    // Find least-loaded resource of given type
    HardwareResource* find_available_resource(ResourceType type, Cycle at_cycle);

    // Calculate transfer size from instruction
    Size get_transfer_size(const DMInstruction& instr) const;

    // Get dependency completion cycle
    Cycle get_dependency_cycle(const DMInstruction& instr) const;

    // Resource allocation strategies
    uint8_t select_dma_channel(MatrixID matrix, TileCoord tile) const;
    uint8_t select_block_mover(uint8_t l3_tile_id) const;
    uint8_t select_streamer(uint8_t l2_bank_id) const;
};

// ============================================================================
// Timeline Formatter
// ============================================================================

/**
 * @brief Formats execution timeline for display
 */
class TimelineFormatter {
public:
    /**
     * @brief Generate ASCII Gantt chart
     */
    static std::string format_gantt(
        const std::vector<ScheduledOp>& ops,
        const ResourceConfig& config,
        Cycle total_cycles,
        size_t width = 80);

    /**
     * @brief Generate resource occupancy table
     */
    static std::string format_occupancy_table(
        const std::vector<ScheduledOp>& ops,
        const ResourceConfig& config,
        Cycle total_cycles);

    /**
     * @brief Generate cycle-by-cycle activity view
     */
    static std::string format_cycle_view(
        const std::vector<ScheduledOp>& ops,
        const ResourceConfig& config,
        Cycle start_cycle,
        Cycle end_cycle);
};

} // namespace sw::kpu::isa
