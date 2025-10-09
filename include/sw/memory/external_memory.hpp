#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <stdexcept>
#include <iomanip>

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
#include <sw/memory/sparse_memory.hpp>

namespace sw::kpu {

// External memory component - manages its own memory model
// Now supports both dense (vector-based) and sparse (memory-mapped) modes
class KPU_API ExternalMemory {
public:
    /**
     * @brief Memory backend type
     */
    enum class BackendType {
        Dense,   ///< Dense allocation using std::vector (old behavior)
        Sparse   ///< Sparse allocation using memory mapping (new, efficient)
    };

    /**
     * @brief Configuration for external memory
     */
    struct Config {
        Size capacity_mb;           ///< Capacity in megabytes
        Size bandwidth_gbps;        ///< Bandwidth in Gbps
        BackendType backend;        ///< Backend type to use
        bool auto_backend;          ///< Auto-select backend based on size

        Config(Size capacity = 1024, Size bandwidth = 100)
            : capacity_mb(capacity)
            , bandwidth_gbps(bandwidth)
            , backend(BackendType::Dense)
            , auto_backend(true) {}
    };

    /**
     * @brief Statistics for external memory
     */
    struct Stats {
        Size capacity_bytes;        ///< Total capacity
        Size resident_bytes;        ///< Actual physical memory used (for sparse)
        Size bandwidth_bps;         ///< Bandwidth in bytes per second
        Cycle last_access_cycle;    ///< Last access cycle
        BackendType backend;        ///< Active backend type
        double utilization;         ///< Memory utilization (0.0 - 1.0)

        Stats() : capacity_bytes(0), resident_bytes(0), bandwidth_bps(0),
                  last_access_cycle(0), backend(BackendType::Dense),
                  utilization(0.0) {}
    };

private:
    // Backend storage (only one will be active)
    std::vector<std::uint8_t> dense_storage_;  // Dense backend
    std::unique_ptr<memory::SparseMemory> sparse_storage_;  // Sparse backend

    BackendType active_backend_;
    Size capacity_;
    Size bandwidth_bytes_per_cycle_;
    mutable Cycle last_access_cycle_;

    // Threshold for auto-selecting sparse backend (in MB)
    static constexpr Size SPARSE_THRESHOLD_MB = 1024;  // Use sparse for >= 1GB

    void initialize_backend(const Config& config);

public:
    /**
     * @brief Construct external memory with default configuration
     * @param capacity_mb Capacity in megabytes (default: 1024)
     * @param bandwidth_gbps Bandwidth in Gbps (default: 100)
     */
    explicit ExternalMemory(Size capacity_mb = 1024, Size bandwidth_gbps = 100);

    /**
     * @brief Construct external memory with full configuration
     * @param config Configuration parameters
     */
    explicit ExternalMemory(const Config& config);

    ~ExternalMemory() = default;

    // Disable copying (contains unique_ptr)
    ExternalMemory(const ExternalMemory&) = delete;
    ExternalMemory& operator=(const ExternalMemory&) = delete;

    // Enable moving
    ExternalMemory(ExternalMemory&&) noexcept = default;
    ExternalMemory& operator=(ExternalMemory&&) noexcept = default;

    // Core memory operations
    void read(Address addr, void* data, Size size);
    void write(Address addr, const void* data, Size size);
    bool is_ready() const;

    // Configuration and status
    Size get_capacity() const { return capacity_; }
    Size get_bandwidth() const { return bandwidth_bytes_per_cycle_; }
    BackendType get_backend() const { return active_backend_; }
    void reset();

    // Statistics
    Cycle get_last_access_cycle() const { return last_access_cycle_; }
    Stats get_stats() const;

    /**
     * @brief Get human-readable backend name
     * @return Backend name string
     */
    const char* get_backend_name() const;

    /**
     * @brief Check if using sparse backend
     * @return true if sparse, false if dense
     */
    bool is_sparse() const { return active_backend_ == BackendType::Sparse; }
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
