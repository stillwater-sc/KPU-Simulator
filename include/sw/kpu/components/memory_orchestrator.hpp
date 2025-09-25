#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>
#include <queue>
#include <array>
#include <mutex>
#include <atomic>
#include <unordered_map>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251) // DLL interface warnings
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

#include <sw/concepts.hpp>

namespace sw::kpu {

// Forward declarations
class BlockMover;
class Streamer;

/**
 * MemoryOrchestrator: Multi-bank buffer memory supporting EDDO (Explicit Decoupled Data Orchestration)
 *
 * NOTE: This is NOT the original "Buffet" from Google's paper, which uses Fill/Read/Update/Shrink
 * for sparse tensor operations. This is a dense memory orchestrator using Prefetch/Compute/Writeback/Sync
 * phases for general-purpose memory hierarchy coordination.
 *
 * EDDO separates control flow from data flow for efficient pipelining and overlap of:
 * - Dense matrix operations (matrix multiplication, convolution)
 * - Multi-stage pipeline coordination
 * - Cross-component workflow orchestration
 *
 * For sparse tensor operations, see SparseBuffet which implements the true Buffet FSM.
 */
class KPU_API MemoryOrchestrator {
public:
    // EDDO Operation Types
    enum class EDDOPhase {
        PREFETCH,      // Asynchronously prefetch data into buffer banks
        COMPUTE,       // Allow compute access to buffered data
        WRITEBACK,     // Write results back to memory hierarchy
        SYNC          // Synchronization barrier between phases
    };

    // Memory Bank Access Patterns for EDDO
    enum class AccessPattern {
        SEQUENTIAL,    // Sequential access within bank
        STRIDED,       // Strided access pattern
        RANDOM,        // Random access (worst case for caching)
        BROADCAST      // One-to-many distribution
    };

    // Memory Bank Configuration
    struct BankConfig {
        Size bank_size_kb;          // Size per bank in KB
        Size cache_line_size;       // Cache line size (typically 64 bytes)
        Size num_ports;             // Number of concurrent access ports
        AccessPattern access_pattern; // Expected access pattern for optimization
        bool enable_prefetch;       // Enable hardware prefetching
    };

    // EDDO Orchestration Command
    struct EDDOCommand {
        EDDOPhase phase;
        size_t bank_id;

        // Memory addressing
        Address source_addr;        // Source address (for PREFETCH/WRITEBACK)
        Address dest_addr;          // Destination address
        Size transfer_size;         // Size in bytes

        // Orchestration control
        size_t sequence_id;         // For ordering dependencies
        std::vector<size_t> dependencies; // Commands that must complete first

        // Integration with data movers
        size_t block_mover_id;      // Which BlockMover to use (-1 if none)
        size_t streamer_id;         // Which Streamer to use (-1 if none)

        // Completion notification
        std::function<void(const EDDOCommand&)> completion_callback;
    };

    // Per-bank state tracking
    struct BankState {
        std::vector<std::uint8_t> data;     // Bank memory storage
        Size capacity;                      // Total capacity in bytes
        Size current_occupancy;             // Current data occupancy

        // Access tracking
        std::atomic<bool> is_reading;       // Currently serving read requests
        std::atomic<bool> is_writing;       // Currently serving write requests

        // EDDO phase tracking
        EDDOPhase current_phase;
        size_t active_sequence_id;

        // Performance counters
        std::atomic<size_t> read_accesses;
        std::atomic<size_t> write_accesses;
        std::atomic<size_t> cache_hits;
        std::atomic<size_t> cache_misses;
    };

private:
    // Configuration
    size_t num_banks;
    std::vector<BankConfig> bank_configs;
    std::vector<std::unique_ptr<BankState>> bank_states;
    size_t orchestrator_id;

    // EDDO command orchestration
    std::queue<EDDOCommand> command_queue;
    std::unordered_map<size_t, EDDOCommand> active_commands;  // sequence_id -> command
    std::unordered_map<size_t, std::vector<size_t>> dependency_graph; // sequence_id -> dependents

    // Thread synchronization (for thread-safe operation)
    mutable std::mutex bank_mutex;
    mutable std::mutex command_mutex;

    // Integration with data movers
    std::vector<BlockMover*> block_movers;
    std::vector<Streamer*> streamers;

    // Sequence ID generator
    size_t next_sequence_id;

    // Internal orchestration methods
    bool can_execute_command(const EDDOCommand& cmd) const;
    void execute_prefetch_command(const EDDOCommand& cmd);
    void execute_compute_command(const EDDOCommand& cmd);
    void execute_writeback_command(const EDDOCommand& cmd);
    void execute_sync_command(const EDDOCommand& cmd);
    void complete_command(const EDDOCommand& cmd);
    void update_dependencies(size_t completed_sequence_id);

    // Bank management
    bool is_bank_available(size_t bank_id, EDDOPhase phase) const;
    void transition_bank_phase(size_t bank_id, EDDOPhase new_phase, size_t sequence_id);

    // Address validation and mapping
    bool validate_bank_access(size_t bank_id, Address addr, Size size) const;
    Address map_to_bank_address(size_t bank_id, Address global_addr) const;

public:
    explicit MemoryOrchestrator(size_t orchestrator_id, size_t num_banks = 4,
                   const BankConfig& default_config = {64, 64, 2, AccessPattern::SEQUENTIAL, true});
    ~MemoryOrchestrator() = default;

    // Custom copy/move for vector compatibility
    MemoryOrchestrator(const MemoryOrchestrator& other);
    MemoryOrchestrator& operator=(const MemoryOrchestrator& other);
    MemoryOrchestrator(MemoryOrchestrator&&) = delete;
    MemoryOrchestrator& operator=(MemoryOrchestrator&&) = delete;

    // Configuration and initialization
    void configure_bank(size_t bank_id, const BankConfig& config);
    void register_block_mover(BlockMover* mover);
    void register_streamer(Streamer* streamer);

    // Core memory operations (direct access)
    void read(size_t bank_id, Address addr, void* data, Size size);
    void write(size_t bank_id, Address addr, const void* data, Size size);
    bool is_ready(size_t bank_id) const;

    // EDDO Orchestration Interface
    void enqueue_eddo_command(const EDDOCommand& cmd);
    bool process_eddo_commands();  // Process one cycle of EDDO commands
    size_t get_pending_commands() const;

    // Advanced EDDO patterns
    void orchestrate_double_buffer(size_t bank_a, size_t bank_b,
                                  Address src_addr, Size transfer_size);
    void orchestrate_pipeline_stage(size_t input_bank, size_t output_bank,
                                   const std::function<void()>& compute_func);

    // Status and performance monitoring
    bool is_busy() const;
    bool is_bank_busy(size_t bank_id) const;
    EDDOPhase get_bank_phase(size_t bank_id) const;

    // Performance metrics
    struct PerformanceMetrics {
        size_t total_read_accesses;
        size_t total_write_accesses;
        size_t total_cache_hits;
        size_t total_cache_misses;
        double average_bank_utilization;
        size_t completed_eddo_commands;
    };
    PerformanceMetrics get_performance_metrics() const;

    // Configuration queries
    size_t get_orchestrator_id() const { return orchestrator_id; }
    size_t get_num_banks() const { return num_banks; }
    Size get_bank_capacity(size_t bank_id) const;
    Size get_bank_occupancy(size_t bank_id) const;

    // Reset and cleanup
    void reset();
    void flush_all_banks();
    void abort_pending_commands();
};

// Helper class for EDDO workflow construction
class KPU_API EDDOWorkflowBuilder {
private:
    std::vector<MemoryOrchestrator::EDDOCommand> commands;
    size_t next_sequence_id;

public:
    EDDOWorkflowBuilder() : next_sequence_id(0) {}

    // Workflow construction methods
    EDDOWorkflowBuilder& prefetch(size_t bank_id, Address src_addr, Address dest_addr, Size size);
    EDDOWorkflowBuilder& compute(size_t bank_id, const std::function<void()>& compute_func);
    EDDOWorkflowBuilder& writeback(size_t bank_id, Address src_addr, Address dest_addr, Size size);
    EDDOWorkflowBuilder& sync();
    EDDOWorkflowBuilder& depend_on(size_t dependency_sequence_id);

    // Build and execute
    std::vector<MemoryOrchestrator::EDDOCommand> build();
    void execute_on(MemoryOrchestrator& orchestrator);
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif