#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>
#include <optional>

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
#include <sw/memory/external_memory.hpp>
#include <sw/kpu/components/scratchpad.hpp>
#include <sw/trace/trace_logger.hpp>

namespace sw::kpu {

// Forward declarations
class L3Tile;
class L2Bank;

// DMA Engine for data movement between memory hierarchies
class KPU_API DMAEngine {
public:
    enum class MemoryType {
        HOST_MEMORY,      // Host DDR
        EXTERNAL,         // KPU memory banks (GDDR6)
        L3_TILE,          // L3 cache tiles
        L2_BANK,          // L2 cache banks
        SCRATCHPAD        // L1 scratchpad
    };

    struct Transfer {
        MemoryType src_type;
        size_t src_id;
        Address src_addr;
        MemoryType dst_type;
        size_t dst_id;
        Address dst_addr;
        Size size;
        std::function<void()> completion_callback;

        // Cycle-based timing
        trace::CycleCount start_cycle;
        trace::CycleCount end_cycle;
        uint64_t transaction_id;  // For trace correlation
    };

private:
    std::vector<Transfer> transfer_queue;  // Dynamically managed resource
    bool is_active;
    size_t engine_id;  // For debugging/identification

    // Multi-cycle timing state (like BlockMover)
    trace::CycleCount cycles_remaining;  // Cycles left for current transfer
    std::vector<uint8_t> transfer_buffer;  // Buffer for current transfer data

    // Tracing support
    bool tracing_enabled_;
    trace::TraceLogger* trace_logger_;
    double clock_freq_ghz_;  // Clock frequency for bandwidth calculations
    double bandwidth_gb_s_;  // Theoretical bandwidth in GB/s

    // Current cycle (for timing)
    trace::CycleCount current_cycle_;

public:
    explicit DMAEngine(size_t engine_id = 0, double clock_freq_ghz = 1.0, double bandwidth_gb_s = 100.0);
    ~DMAEngine() = default;

    // Enable/disable tracing
    void enable_tracing(bool enabled = true, trace::TraceLogger* logger = nullptr) {
        tracing_enabled_ = enabled;
        if (logger) trace_logger_ = logger;
    }

    // Set current cycle (called by system clock/orchestrator)
    void set_current_cycle(trace::CycleCount cycle) {
        current_cycle_ = cycle;
    }

    trace::CycleCount get_current_cycle() const {
        return current_cycle_;
    }

    // Transfer operations - now cycle-aware
    void enqueue_transfer(MemoryType src_type, size_t src_id, Address src_addr,
                         MemoryType dst_type, size_t dst_id, Address dst_addr,
                         Size size, std::function<void()> callback = nullptr);

    // Process transfers with full memory hierarchy access
    bool process_transfers(std::vector<ExternalMemory>& memory_banks,
                          std::vector<L3Tile>& l3_tiles,
                          std::vector<L2Bank>& l2_banks,
                          std::vector<Scratchpad>& scratchpads);

    bool is_busy() const { return is_active || !transfer_queue.empty(); }
    void reset();

    // Status and identification
    size_t get_engine_id() const { return engine_id; }
    size_t get_queue_size() const { return transfer_queue.size(); }
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif