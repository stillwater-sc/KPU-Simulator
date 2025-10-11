#pragma once

#include <sw/trace/trace_entry.hpp>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>
#include <algorithm>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251)
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

namespace sw::trace {

// Filter function type for querying traces
using TraceFilter = std::function<bool(const TraceEntry&)>;

// Thread-safe trace logger for KPU simulation
class KPU_API TraceLogger {
public:
    // Singleton pattern for global trace collection
    static TraceLogger& instance() {
        static TraceLogger logger;
        return logger;
    }

    // Configuration
    struct Config {
        bool enabled = true;              // Enable/disable tracing
        size_t buffer_reserve = 100000;   // Pre-allocate buffer space
        bool record_all = true;           // Record all transactions (if false, use filters)

        Config() = default;
    };

    // Initialize with configuration
    void initialize(const Config& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = config;
        if (config_.buffer_reserve > 0) {
            traces_.reserve(config_.buffer_reserve);
        }
    }

    // Enable/disable tracing dynamically
    void set_enabled(bool enabled) {
        enabled_.store(enabled, std::memory_order_relaxed);
    }

    bool is_enabled() const {
        return enabled_.load(std::memory_order_relaxed);
    }

    // Log a trace entry (move semantics for efficiency)
    void log(TraceEntry&& entry) {
        if (!is_enabled()) return;

        std::lock_guard<std::mutex> lock(mutex_);
        traces_.emplace_back(std::move(entry));
        transaction_counter_++;
    }

    // Log a trace entry (copy)
    void log(const TraceEntry& entry) {
        if (!is_enabled()) return;

        std::lock_guard<std::mutex> lock(mutex_);
        traces_.push_back(entry);
        transaction_counter_++;
    }

    // Get next transaction ID (thread-safe)
    uint64_t next_transaction_id() {
        return transaction_counter_.fetch_add(1, std::memory_order_relaxed);
    }

    // Query operations (thread-safe)
    std::vector<TraceEntry> get_all_traces() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return traces_;
    }

    // Get filtered traces
    std::vector<TraceEntry> get_filtered_traces(const TraceFilter& filter) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<TraceEntry> result;
        std::copy_if(traces_.begin(), traces_.end(), std::back_inserter(result), filter);
        return result;
    }

    // Get traces for a specific component
    std::vector<TraceEntry> get_component_traces(ComponentType type, uint32_t id) const {
        return get_filtered_traces([type, id](const TraceEntry& entry) {
            return entry.component_type == type && entry.component_id == id;
        });
    }

    // Get traces within a cycle range
    std::vector<TraceEntry> get_traces_in_range(CycleCount start, CycleCount end) const {
        return get_filtered_traces([start, end](const TraceEntry& entry) {
            return entry.cycle_issue >= start && entry.cycle_issue <= end;
        });
    }

    // Get traces by transaction type
    std::vector<TraceEntry> get_transaction_type_traces(TransactionType type) const {
        return get_filtered_traces([type](const TraceEntry& entry) {
            return entry.transaction_type == type;
        });
    }

    // Statistics
    size_t get_trace_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return traces_.size();
    }

    CycleCount get_min_cycle() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (traces_.empty()) return 0;
        return std::min_element(traces_.begin(), traces_.end(),
            [](const TraceEntry& a, const TraceEntry& b) {
                return a.cycle_issue < b.cycle_issue;
            })->cycle_issue;
    }

    CycleCount get_max_cycle() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (traces_.empty()) return 0;
        return std::max_element(traces_.begin(), traces_.end(),
            [](const TraceEntry& a, const TraceEntry& b) {
                return a.cycle_complete < b.cycle_complete;
            })->cycle_complete;
    }

    // Clear all traces
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        traces_.clear();
        transaction_counter_.store(0, std::memory_order_relaxed);
    }

    // Reset to initial state
    void reset() {
        clear();
        config_ = Config();
    }

private:
    // Private constructor for singleton
    TraceLogger() : enabled_(true), transaction_counter_(0) {
        config_ = Config();
    }

    // Prevent copying
    TraceLogger(const TraceLogger&) = delete;
    TraceLogger& operator=(const TraceLogger&) = delete;

    // Thread synchronization
    mutable std::mutex mutex_;

    // Configuration
    Config config_;

    // Enable flag (atomic for fast check)
    std::atomic<bool> enabled_;

    // Transaction ID counter (atomic for thread-safe ID generation)
    std::atomic<uint64_t> transaction_counter_;

    // Trace storage
    std::vector<TraceEntry> traces_;
};

// RAII helper for automatically completing transactions
class KPU_API ScopedTrace {
public:
    ScopedTrace(TraceEntry&& entry, TraceLogger& logger = TraceLogger::instance())
        : entry_(std::move(entry))
        , logger_(logger)
        , logged_(false)
    {
        logger_.log(entry_);
        logged_ = true;
    }

    // Complete the transaction when scope exits
    ~ScopedTrace() {
        // Note: In a real implementation, we'd need to update the logged entry
        // For now, this is a placeholder for the pattern
    }

    // Mark as completed manually
    void complete(CycleCount cycle, TransactionStatus status = TransactionStatus::COMPLETED) {
        entry_.complete(cycle, status);
        // Re-log the updated entry
        logger_.log(entry_);
    }

    // Get the transaction ID
    uint64_t get_transaction_id() const {
        return entry_.transaction_id;
    }

private:
    TraceEntry entry_;
    TraceLogger& logger_;
    bool logged_;
};

} // namespace sw::trace

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
