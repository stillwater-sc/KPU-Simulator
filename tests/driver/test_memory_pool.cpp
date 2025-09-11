#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <sw/driver/memory_manager.hpp>
#include <vector>
#include <set>
#include <chrono>
#include <random>

using namespace sw::driver;

#ifdef DRVIER

TEST_CASE("Memory pool basic operations", "[memory][pool]") {
    constexpr size_t pool_size = 4096;
    constexpr size_t block_size = 64;
    
    MemoryPool pool(pool_size, block_size);
    
    SECTION("Pool initialization") {
        REQUIRE(pool.get_block_size() == block_size);
        REQUIRE(pool.get_total_blocks() == pool_size / block_size);
        REQUIRE(pool.get_available_blocks() == pool.get_total_blocks());
        REQUIRE(pool.get_used_blocks() == 0);
    }
    
    SECTION("Single block allocation") {
        auto ptr = pool.allocate();
        
        REQUIRE(ptr != nullptr);
        REQUIRE(pool.get_used_blocks() == 1);
        REQUIRE(pool.get_available_blocks() == pool.get_total_blocks() - 1);
        REQUIRE(pool.is_from_pool(ptr));
        
        pool.deallocate(ptr);
        
        REQUIRE(pool.get_used_blocks() == 0);
        REQUIRE(pool.get_available_blocks() == pool.get_total_blocks());
    }
    
    SECTION("Multiple block allocation") {
        std::vector<void*> ptrs;
        constexpr size_t num_blocks = 10;
        
        for (size_t i = 0; i < num_blocks; ++i) {
            auto ptr = pool.allocate();
            REQUIRE(ptr != nullptr);
            ptrs.push_back(ptr);
        }
        
        REQUIRE(pool.get_used_blocks() == num_blocks);
        REQUIRE(pool.get_available_blocks() == pool.get_total_blocks() - num_blocks);
        
        // Verify all pointers are distinct and aligned to block boundaries
        std::set<void*> unique_ptrs(ptrs.begin(), ptrs.end());
        REQUIRE(unique_ptrs.size() == num_blocks);
        
        for (auto ptr : ptrs) {
            REQUIRE(pool.is_from_pool(ptr));
            REQUIRE(reinterpret_cast<uintptr_t>(ptr) % block_size == 0);
        }
        
        // Clean up
        for (auto ptr : ptrs) {
            pool.deallocate(ptr);
        }
        
        REQUIRE(pool.get_used_blocks() == 0);
    }
}

TEST_CASE("Memory pool exhaustion", "[memory][pool][exhaustion]") {
    constexpr size_t pool_size = 1024;
    constexpr size_t block_size = 64;
    constexpr size_t max_blocks = pool_size / block_size;
    
    MemoryPool pool(pool_size, block_size);
    std::vector<void*> ptrs;
    
    SECTION("Allocate all blocks") {
        // Allocate all available blocks
        for (size_t i = 0; i < max_blocks; ++i) {
            auto ptr = pool.allocate();
            REQUIRE(ptr != nullptr);
            ptrs.push_back(ptr);
        }
        
        REQUIRE(pool.get_used_blocks() == max_blocks);
        REQUIRE(pool.get_available_blocks() == 0);
        REQUIRE(pool.is_full());
        
        // Next allocation should fail
        auto ptr = pool.allocate();
        REQUIRE(ptr == nullptr);
        
        // Clean up
        for (auto p : ptrs) {
            pool.deallocate(p);
        }
    }
    
    SECTION("Pool capacity recovery") {
        // Fill the pool
        for (size_t i = 0; i < max_blocks; ++i) {
            ptrs.push_back(pool.allocate());
        }
        
        REQUIRE(pool.is_full());
        
        // Free half the blocks
        for (size_t i = 0; i < max_blocks / 2; ++i) {
            pool.deallocate(ptrs[i]);
        }
        
        REQUIRE_FALSE(pool.is_full());
        REQUIRE(pool.get_available_blocks() == max_blocks / 2);
        
        // Should be able to allocate again
        auto new_ptr = pool.allocate();
        REQUIRE(new_ptr != nullptr);
        
        // Clean up remaining
        pool.deallocate(new_ptr);
        for (size_t i = max_blocks / 2; i < max_blocks; ++i) {
            pool.deallocate(ptrs[i]);
        }
    }
}

TEST_CASE("Memory pool allocation patterns", "[memory][pool][patterns]") {
    constexpr size_t pool_size = 2048;
    constexpr size_t block_size = 128;
    
    MemoryPool pool(pool_size, block_size);
    
    SECTION("LIFO allocation pattern") {
        std::vector<void*> ptrs;
        
        // Allocate several blocks
        for (int i = 0; i < 5; ++i) {
            ptrs.push_back(pool.allocate());
        }
        
        // Free in reverse order (LIFO)
        for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it) {
            pool.deallocate(*it);
        }
        
        REQUIRE(pool.get_used_blocks() == 0);
    }
    
    SECTION("FIFO allocation pattern") {
        std::vector<void*> ptrs;
        
        // Allocate several blocks
        for (int i = 0; i < 5; ++i) {
            ptrs.push_back(pool.allocate());
        }
        
        // Free in same order (FIFO)
        for (auto ptr : ptrs) {
            pool.deallocate(ptr);
        }
        
        REQUIRE(pool.get_used_blocks() == 0);
    }
    
    SECTION("Random allocation/deallocation") {
        std::vector<void*> ptrs;
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Random allocation/deallocation pattern
        for (int i = 0; i < 100; ++i) {
            if (ptrs.empty() || (ptrs.size() < 8 && gen() % 2)) {
                // Allocate
                auto ptr = pool.allocate();
                if (ptr != nullptr) {
                    ptrs.push_back(ptr);
                }
            } else {
                // Deallocate random block
                auto idx = gen() % ptrs.size();
                pool.deallocate(ptrs[idx]);
                ptrs.erase(ptrs.begin() + idx);
            }
        }
        
        // Clean up remaining
        for (auto ptr : ptrs) {
            pool.deallocate(ptr);
        }
        
        REQUIRE(pool.get_used_blocks() == 0);
    }
}

TEST_CASE("Memory pool error handling", "[memory][pool][error]") {
    constexpr size_t pool_size = 1024;
    constexpr size_t block_size = 64;
    
    MemoryPool pool(pool_size, block_size);
    
    SECTION("Double deallocation") {
        auto ptr = pool.allocate();
        REQUIRE(ptr != nullptr);
        
        pool.deallocate(ptr);
        REQUIRE(pool.get_used_blocks() == 0);
        
        // Double deallocation should be detected
        REQUIRE_THROWS_AS(pool.deallocate(ptr), std::invalid_argument);
    }
    
    SECTION("Deallocate foreign pointer") {
        int stack_var = 42;
        void* foreign_ptr = &stack_var;
        
        REQUIRE_FALSE(pool.is_from_pool(foreign_ptr));
        REQUIRE_THROWS_AS(pool.deallocate(foreign_ptr), std::invalid_argument);
    }
    
    SECTION("Deallocate null pointer") {
        // Should handle null gracefully
        REQUIRE_NOTHROW(pool.deallocate(nullptr));
    }
    
    SECTION("Deallocate unaligned pool pointer") {
        auto ptr = pool.allocate();
        REQUIRE(ptr != nullptr);
        
        // Create misaligned pointer within pool range
        auto misaligned = static_cast<char*>(ptr) + 1;
        
        REQUIRE_THROWS_AS(pool.deallocate(misaligned), std::invalid_argument);
        
        // Clean up properly
        pool.deallocate(ptr);
    }
}

TEST_CASE("Memory pool performance characteristics", "[memory][pool][performance]") {
    constexpr size_t pool_size = 65536;  // 64KB
    constexpr size_t block_size = 64;
    constexpr size_t num_operations = 1000;
    
    MemoryPool pool(pool_size, block_size);
    
    SECTION("Allocation performance") {
        std::vector<void*> ptrs;
        ptrs.reserve(num_operations);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < num_operations; ++i) {
            auto ptr = pool.allocate();
            if (ptr) ptrs.push_back(ptr);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Performance should be consistent (pool allocation is O(1))
        REQUIRE(duration.count() < 10000);  // Should complete in less than 10ms
        
        // Clean up
        for (auto ptr : ptrs) {
            pool.deallocate(ptr);
        }
    }
    
    SECTION("Fragmentation resistance") {
        std::vector<void*> ptrs;
        
        // Allocate many blocks
        for (size_t i = 0; i < 100; ++i) {
            ptrs.push_back(pool.allocate());
        }
        
        // Free every other block (create fragmentation)
        for (size_t i = 0; i < ptrs.size(); i += 2) {
            pool.deallocate(ptrs[i]);
            ptrs[i] = nullptr;
        }
        
        // Should still be able to allocate in fragmented state
        size_t successful_allocs = 0;
        for (size_t i = 0; i < 50; ++i) {
            auto ptr = pool.allocate();
            if (ptr) {
                ptrs.push_back(ptr);
                ++successful_allocs;
            }
        }
        
        REQUIRE(successful_allocs > 0);
        
        // Clean up
        for (auto ptr : ptrs) {
            if (ptr) pool.deallocate(ptr);
        }
    }
}

#endif