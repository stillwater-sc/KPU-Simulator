#pragma once

#include <cstddef>
#include <stdexcept>
#include <cstdint>
#include <vector>

namespace sw::driver {

/**
 * @brief Basic memory manager for Stillwater System simulator
 * 
 * Provides basic allocation and deallocation services with
 * tracking and validation capabilities.
 */
class MemoryManager {
public:
    MemoryManager() = default;
    ~MemoryManager() = default;
    
    // Non-copyable, movable
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) = default;
    MemoryManager& operator=(MemoryManager&&) = default;
    
    /**
     * @brief Allocate memory of specified size
     * @param size Number of bytes to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     */
    void* allocate(size_t size);
    
    /**
     * @brief Deallocate previously allocated memory
     * @param ptr Pointer to memory to deallocate
     * @throws std::invalid_argument if ptr is invalid
     */
    void deallocate(void* ptr);
    
    /**
     * @brief Check if a pointer is valid (was allocated by this manager)
     * @param ptr Pointer to check
     * @return True if pointer is valid, false otherwise
     */
    bool is_valid_address(void* ptr) const;
    
    /**
     * @brief Get the size of an allocation
     * @param ptr Pointer to allocated memory
     * @return Size of the allocation in bytes
     * @throws std::invalid_argument if ptr is invalid
     */
    size_t get_allocation_size(void* ptr) const;
    
    /**
     * @brief Get current number of active allocations
     * @return Number of currently allocated blocks
     */
    size_t get_allocation_count() const noexcept { return allocation_count_; }
    
    /**
     * @brief Get current total allocated bytes
     * @return Total bytes currently allocated
     */
    size_t get_allocated_bytes() const noexcept { return allocated_bytes_; }
    
    /**
     * @brief Get peak allocated bytes since creation
     * @return Maximum bytes that were allocated simultaneously
     */
    size_t get_peak_allocated_bytes() const noexcept { return peak_allocated_bytes_; }
    
    /**
     * @brief Reset peak tracking
     */
    void reset_peak_tracking() noexcept { peak_allocated_bytes_ = allocated_bytes_; }

private:
    size_t allocation_count_{0};
    size_t allocated_bytes_{0};
    size_t peak_allocated_bytes_{0};
    
    void update_statistics(size_t size, bool allocating);
};

/**
 * @brief Memory pool for fixed-size block allocation
 * 
 * Provides fast O(1) allocation and deallocation for fixed-size blocks.
 * Useful for high-frequency allocations of similar-sized objects.
 */
class MemoryPool {
public:
    /**
     * @brief Create a memory pool with specified parameters
     * @param pool_size Total size of the pool in bytes
     * @param block_size Size of each block in bytes
     */
    MemoryPool(size_t pool_size, size_t block_size);
    
    /**
     * @brief Destructor - releases pool memory
     */
    ~MemoryPool();
    
    // Non-copyable, movable
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = default;
    MemoryPool& operator=(MemoryPool&&) = default;
    
    /**
     * @brief Allocate a block from the pool
     * @return Pointer to allocated block, or nullptr if pool is full
     */
    void* allocate();
    
    /**
     * @brief Deallocate a block back to the pool
     * @param ptr Pointer to block to deallocate
     * @throws std::invalid_argument if ptr is not from this pool
     */
    void deallocate(void* ptr);
    
    /**
     * @brief Check if pointer belongs to this pool
     * @param ptr Pointer to check
     * @return True if pointer is from this pool
     */
    bool is_from_pool(void* ptr) const;
    
    /**
     * @brief Get the block size for this pool
     * @return Size of each block in bytes
     */
    size_t get_block_size() const noexcept { return block_size_; }
    
    /**
     * @brief Get total number of blocks in pool
     * @return Total block capacity
     */
    size_t get_total_blocks() const noexcept { return total_blocks_; }
    
    /**
     * @brief Get number of currently used blocks
     * @return Number of allocated blocks
     */
    size_t get_used_blocks() const noexcept { return used_blocks_; }
    
    /**
     * @brief Get number of available blocks
     * @return Number of free blocks
     */
    size_t get_available_blocks() const noexcept { return total_blocks_ - used_blocks_; }
    
    /**
     * @brief Check if pool is full
     * @return True if no blocks are available
     */
    bool is_full() const noexcept { return used_blocks_ == total_blocks_; }
    
    /**
     * @brief Check if pool is empty
     * @return True if no blocks are allocated
     */
    bool is_empty() const noexcept { return used_blocks_ == 0; }

private:
    void* pool_memory_{nullptr};
    size_t pool_size_;
    size_t block_size_;
    size_t total_blocks_;
    size_t used_blocks_{0};
    
    // Free list management
    void* free_list_head_{nullptr};
    
    void initialize_free_list();
    bool is_valid_pool_address(void* ptr) const;
};

/**
 * @brief Aligned memory manager for SIMD and cache-line optimization
 * 
 * Provides memory allocation with specific alignment requirements,
 * useful for SIMD operations and cache optimization.
 */
class AlignedMemoryManager {
public:
    AlignedMemoryManager() = default;
    ~AlignedMemoryManager() = default;
    
    // Non-copyable, movable
    AlignedMemoryManager(const AlignedMemoryManager&) = delete;
    AlignedMemoryManager& operator=(const AlignedMemoryManager&) = delete;
    AlignedMemoryManager(AlignedMemoryManager&&) = default;
    AlignedMemoryManager& operator=(AlignedMemoryManager&&) = default;
    
    /**
     * @brief Allocate aligned memory
     * @param size Number of bytes to allocate
     * @param alignment Required alignment (must be power of 2)
     * @return Pointer to aligned memory, or nullptr on failure
     * @throws std::invalid_argument if alignment is invalid
     */
    void* allocate_aligned(size_t size, size_t alignment);
    
    /**
     * @brief Deallocate aligned memory
     * @param ptr Pointer to memory allocated with allocate_aligned
     * @throws std::invalid_argument if ptr is invalid
     */
    void deallocate_aligned(void* ptr);
    
    /**
     * @brief Check if pointer is from this allocator
     * @param ptr Pointer to check
     * @return True if pointer was allocated by this manager
     */
    bool is_valid_aligned_address(void* ptr) const;
    
    /**
     * @brief Get allocation info for a pointer
     * @param ptr Pointer to query
     * @return Pair of (size, alignment) for the allocation
     * @throws std::invalid_argument if ptr is invalid
     */
    std::pair<size_t, size_t> get_allocation_info(void* ptr) const;

private:
    struct AlignedAllocation {
        void* original_ptr;
        void* aligned_ptr;
        size_t size;
        size_t alignment;
    };
    
    std::vector<AlignedAllocation> allocations_;
    
    static bool is_power_of_two(size_t value) noexcept;
    static void* align_pointer(void* ptr, size_t alignment) noexcept;
};

} // namespace sw::driver