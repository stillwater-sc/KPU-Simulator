#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

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

namespace sw::kpu::memory {

/**
 * @brief Cross-platform memory mapping abstraction
 *
 * Provides a unified interface for memory-mapped files and anonymous mappings
 * across Windows, Linux, and MacOS. Supports sparse memory allocation where
 * the OS only commits physical pages on first write (copy-on-write semantics).
 *
 * Design philosophy:
 * - Virtual address space reservation without physical memory commitment
 * - On-demand page allocation (OS allocates pages on fault)
 * - Automatic cleanup via RAII
 * - Platform-specific optimizations hidden behind common interface
 */
class KPU_API MemoryMap {
public:
    /**
     * @brief Configuration for memory mapping
     */
    struct Config {
        Size size;              ///< Virtual address space size in bytes
        bool read_only;         ///< Read-only mapping (default: false)
        bool populate;          ///< Prefault pages (default: false, for sparse)
        Size page_size;         ///< Page size hint (0 = use system default)

        Config(Size size_bytes = 0)
            : size(size_bytes)
            , read_only(false)
            , populate(false)
            , page_size(0) {}
    };

    /**
     * @brief Statistics for memory mapping
     */
    struct Stats {
        Size virtual_size;      ///< Reserved virtual address space
        Size resident_size;     ///< Actual physical memory committed
        Size page_size;         ///< System page size
        Size page_faults;       ///< Number of page faults (if available)

        Stats() : virtual_size(0), resident_size(0), page_size(0), page_faults(0) {}
    };

private:
    void* base_ptr_;           ///< Base address of mapped region
    Size size_;                ///< Size of mapped region
    bool is_mapped_;           ///< Whether mapping is active
    Size page_size_;           ///< System page size

    // Platform-specific handle
#ifdef _WIN32
    void* file_mapping_handle_; ///< Windows file mapping handle
#else
    int fd_;                    ///< Unix file descriptor (-1 for anonymous)
#endif

    // Platform-specific implementation helpers
    void platform_init();
    void* platform_map(const Config& config);
    void platform_unmap();
    Size platform_get_resident_size() const;

public:
    /**
     * @brief Construct a memory map with specified configuration
     * @param config Memory mapping configuration
     * @throws std::runtime_error if mapping fails
     */
    explicit MemoryMap(const Config& config);

    /**
     * @brief Destructor - automatically unmaps memory
     */
    ~MemoryMap();

    // Disable copying (resource management)
    MemoryMap(const MemoryMap&) = delete;
    MemoryMap& operator=(const MemoryMap&) = delete;

    // Enable moving
    MemoryMap(MemoryMap&& other) noexcept;
    MemoryMap& operator=(MemoryMap&& other) noexcept;

    /**
     * @brief Get base pointer to mapped memory
     * @return Pointer to start of mapped region
     */
    void* data() noexcept { return base_ptr_; }
    const void* data() const noexcept { return base_ptr_; }

    /**
     * @brief Get size of mapped region
     * @return Size in bytes
     */
    Size size() const noexcept { return size_; }

    /**
     * @brief Check if mapping is valid
     * @return true if mapped, false otherwise
     */
    bool is_mapped() const noexcept { return is_mapped_; }

    /**
     * @brief Get system page size
     * @return Page size in bytes
     */
    Size page_size() const noexcept { return page_size_; }

    /**
     * @brief Get memory statistics
     * @return Current memory usage statistics
     */
    Stats get_stats() const;

    /**
     * @brief Advise the OS about expected access patterns
     * @param offset Offset within mapping
     * @param length Length of region
     * @param advice Platform-independent advice hint
     */
    enum class Advice {
        Normal,        ///< No special treatment
        Sequential,    ///< Expect sequential access
        Random,        ///< Expect random access
        WillNeed,      ///< Will need these pages soon (prefetch)
        DontNeed,      ///< Don't need these pages (can discard)
    };
    void advise(Size offset, Size length, Advice advice);

    /**
     * @brief Prefault pages in a region (force allocation)
     * @param offset Offset within mapping
     * @param length Length of region to prefault
     */
    void prefault(Size offset, Size length);

    /**
     * @brief Synchronize mapping to backing store (if file-backed)
     * @param offset Offset within mapping
     * @param length Length of region to sync
     * @param async true for asynchronous sync
     */
    void sync(Size offset, Size length, bool async = false);

    /**
     * @brief Get system page size
     * @return System page size in bytes
     */
    static Size get_system_page_size();

    /**
     * @brief Check if huge pages are available
     * @return true if huge pages supported
     */
    static bool has_huge_pages();
};

/**
 * @brief RAII wrapper for memory-mapped regions with typed access
 *
 * Provides convenient typed access to memory-mapped regions with
 * bounds checking and alignment verification.
 */
template<typename T>
class TypedMemoryMap {
private:
    std::unique_ptr<MemoryMap> map_;
    Size count_;

public:
    explicit TypedMemoryMap(Size count)
        : count_(count) {
        MemoryMap::Config config(count * sizeof(T));
        map_ = std::make_unique<MemoryMap>(config);

        // Verify alignment
        if (reinterpret_cast<std::uintptr_t>(map_->data()) % alignof(T) != 0) {
            throw std::runtime_error("Memory map not properly aligned for type");
        }
    }

    T* data() noexcept {
        return static_cast<T*>(map_->data());
    }

    const T* data() const noexcept {
        return static_cast<const T*>(map_->data());
    }

    Size size() const noexcept { return count_; }
    Size size_bytes() const noexcept { return count_ * sizeof(T); }

    T& operator[](Size index) {
        if (index >= count_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data()[index];
    }

    const T& operator[](Size index) const {
        if (index >= count_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data()[index];
    }

    MemoryMap::Stats get_stats() const {
        return map_->get_stats();
    }

    void advise(MemoryMap::Advice advice) {
        map_->advise(0, map_->size(), advice);
    }
};

} // namespace sw::kpu::memory

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
