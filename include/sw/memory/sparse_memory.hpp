#pragma once

#include <memory>
#include <unordered_set>
#include <mutex>
#include <cstdint>

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
#include <sw/memory/memory_map.hpp>

namespace sw::kpu::memory {

/**
 * @brief Sparse memory manager with on-demand page allocation
 *
 * Provides a large virtual address space with physical memory allocated
 * only for pages that are actually accessed. Ideal for scenarios where:
 * - Address space is very large (e.g., 48-bit addressing = 256TB)
 * - Actual memory usage is sparse (e.g., testing specific regions)
 * - Memory efficiency is critical
 *
 * Features:
 * - Automatic page tracking
 * - Thread-safe access
 * - Statistics on memory usage
 * - Optional zero-initialization on first access
 * - Configurable page size
 */
class KPU_API SparseMemory {
public:
    /**
     * @brief Configuration for sparse memory
     */
    struct Config {
        Size virtual_size;      ///< Total virtual address space (in bytes)
        Size page_size;         ///< Page size (0 = use system default)
        bool zero_on_access;    ///< Zero pages on first access (default: true)
        bool track_pages;       ///< Track accessed pages for statistics (default: true)
        bool thread_safe;       ///< Enable thread-safe operations (default: true)

        Config(Size size = 0)
            : virtual_size(size)
            , page_size(0)
            , zero_on_access(true)
            , track_pages(true)
            , thread_safe(true) {}
    };

    /**
     * @brief Statistics for sparse memory usage
     */
    struct Stats {
        Size virtual_size;      ///< Total virtual address space
        Size resident_size;     ///< Physical memory actually committed
        Size accessed_pages;    ///< Number of pages accessed (if tracked)
        Size page_size;         ///< Page size in bytes
        double utilization;     ///< Resident / Virtual ratio

        Stats() : virtual_size(0), resident_size(0), accessed_pages(0),
                  page_size(0), utilization(0.0) {}
    };

private:
    std::unique_ptr<MemoryMap> map_;
    Config config_;
    mutable std::unique_ptr<std::mutex> mutex_;  // For thread-safe operations
    std::unordered_set<Size> accessed_pages_;    // Track which pages accessed

    // Internal helpers
    Size get_page_index(Address addr) const;
    void ensure_page_committed(Address addr);
    void track_page_access(Address addr);

public:
    /**
     * @brief Construct sparse memory with configuration
     * @param config Sparse memory configuration
     */
    explicit SparseMemory(const Config& config);

    /**
     * @brief Destructor
     */
    ~SparseMemory() = default;

    // Disable copying
    SparseMemory(const SparseMemory&) = delete;
    SparseMemory& operator=(const SparseMemory&) = delete;

    // Enable moving
    SparseMemory(SparseMemory&&) noexcept = default;
    SparseMemory& operator=(SparseMemory&&) noexcept = default;

    /**
     * @brief Read from sparse memory
     * @param addr Address to read from
     * @param data Destination buffer
     * @param size Number of bytes to read
     * @throws std::out_of_range if address out of bounds
     */
    void read(Address addr, void* data, Size size);

    /**
     * @brief Write to sparse memory
     * @param addr Address to write to
     * @param data Source buffer
     * @param size Number of bytes to write
     * @throws std::out_of_range if address out of bounds
     */
    void write(Address addr, const void* data, Size size);

    /**
     * @brief Get raw pointer to memory region
     * @param addr Address within the sparse memory
     * @return Pointer to the address
     * @throws std::out_of_range if address out of bounds
     * @warning Direct pointer access bypasses page tracking
     */
    void* get_pointer(Address addr);
    const void* get_pointer(Address addr) const;

    /**
     * @brief Get statistics on memory usage
     * @return Current memory statistics
     */
    Stats get_stats() const;

    /**
     * @brief Get total virtual size
     * @return Virtual address space size in bytes
     */
    Size size() const { return config_.virtual_size; }

    /**
     * @brief Get page size
     * @return Page size in bytes
     */
    Size page_size() const;

    /**
     * @brief Prefault a range of pages
     * @param addr Starting address
     * @param size Number of bytes
     */
    void prefault(Address addr, Size size);

    /**
     * @brief Clear all data (reset to zero)
     */
    void clear();

    /**
     * @brief Advise the OS about access patterns
     * @param addr Starting address
     * @param size Number of bytes
     * @param advice Access pattern advice
     */
    void advise(Address addr, Size size, MemoryMap::Advice advice);

    /**
     * @brief Check if an address is within bounds
     * @param addr Address to check
     * @return true if valid, false otherwise
     */
    bool is_valid_address(Address addr) const {
        return addr < config_.virtual_size;
    }

    /**
     * @brief Check if an address range is within bounds
     * @param addr Starting address
     * @param size Number of bytes
     * @return true if valid, false otherwise
     */
    bool is_valid_range(Address addr, Size size) const {
        return addr < config_.virtual_size &&
               addr + size <= config_.virtual_size &&
               addr + size >= addr;  // Check for overflow
    }
};

} // namespace sw::kpu::memory

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
