#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <sw/kpu/memory_manager.hpp>
#include <memory>
#include <vector>
#include <cstddef>

using namespace sw::kpu;

TEST_CASE("Basic memory allocation", "[memory][allocation]") {
    MemoryManager memory_mgr;
    
    SECTION("Single allocation") {
        constexpr size_t size = 1024;
        auto ptr = memory_mgr.allocate(size);
        
        REQUIRE(ptr != nullptr);
        REQUIRE(memory_mgr.is_valid_address(ptr));
        REQUIRE(memory_mgr.get_allocation_size(ptr) >= size);
        
        memory_mgr.deallocate(ptr);
    }
    
    SECTION("Multiple allocations") {
        std::vector<void*> ptrs;
        constexpr size_t num_allocs = 10;
        constexpr size_t alloc_size = 512;
        
        for (size_t i = 0; i < num_allocs; ++i) {
            auto ptr = memory_mgr.allocate(alloc_size);
            REQUIRE(ptr != nullptr);
            ptrs.push_back(ptr);
        }
        
        // Verify all allocations are distinct
        for (size_t i = 0; i < ptrs.size(); ++i) {
            for (size_t j = i + 1; j < ptrs.size(); ++j) {
                REQUIRE(ptrs[i] != ptrs[j]);
            }
        }
        
        // Clean up
        for (auto ptr : ptrs) {
            memory_mgr.deallocate(ptr);
        }
    }
    
    SECTION("Zero size allocation") {
        auto ptr = memory_mgr.allocate(0);
        // Implementation-defined behavior: may return nullptr or valid pointer
        if (ptr != nullptr) {
            memory_mgr.deallocate(ptr);
        }
    }
}

TEST_CASE("Memory allocation edge cases", "[memory][allocation][edge]") {
    MemoryManager memory_mgr;
    
    SECTION("Large allocation") {
        constexpr size_t large_size = 1024 * 1024;  // 1MB
        auto ptr = memory_mgr.allocate(large_size);
        
        if (ptr != nullptr) {  // May fail due to memory constraints
            REQUIRE(memory_mgr.is_valid_address(ptr));
            REQUIRE(memory_mgr.get_allocation_size(ptr) >= large_size);
            memory_mgr.deallocate(ptr);
        }
    }
    
    SECTION("Power-of-two sizes") {
        auto size = GENERATE(64, 128, 256, 512, 1024, 2048, 4096);
        
        auto ptr = memory_mgr.allocate(size);
        REQUIRE(ptr != nullptr);
        REQUIRE(memory_mgr.get_allocation_size(ptr) >= size);
        
        memory_mgr.deallocate(ptr);
    }
    
    SECTION("Odd sizes") {
        auto size = GENERATE(33, 127, 333, 777, 1001);
        
        auto ptr = memory_mgr.allocate(size);
        REQUIRE(ptr != nullptr);
        REQUIRE(memory_mgr.get_allocation_size(ptr) >= size);
        
        memory_mgr.deallocate(ptr);
    }
}

TEST_CASE("Memory deallocation", "[memory][allocation][deallocation]") {
    MemoryManager memory_mgr;
    
    SECTION("Valid deallocation") {
        auto ptr = memory_mgr.allocate(1024);
        REQUIRE(ptr != nullptr);
        
        REQUIRE_NOTHROW(memory_mgr.deallocate(ptr));
        REQUIRE_FALSE(memory_mgr.is_valid_address(ptr));
    }
    
    SECTION("Double deallocation") {
        auto ptr = memory_mgr.allocate(1024);
        REQUIRE(ptr != nullptr);
        
        memory_mgr.deallocate(ptr);
        
        // Double deallocation should be detected and handled gracefully
        REQUIRE_THROWS_AS(memory_mgr.deallocate(ptr), std::invalid_argument);
    }
    
    SECTION("Null pointer deallocation") {
        // Deallocating null should be safe (like std::free)
        REQUIRE_NOTHROW(memory_mgr.deallocate(nullptr));
    }
    
    SECTION("Invalid pointer deallocation") {
        int stack_var = 42;
        void* invalid_ptr = &stack_var;
        
        REQUIRE_THROWS_AS(memory_mgr.deallocate(invalid_ptr), std::invalid_argument);
    }
}

TEST_CASE("Memory allocation statistics", "[memory][allocation][stats]") {
    MemoryManager memory_mgr;
    
    SECTION("Track allocations") {
        auto initial_count = memory_mgr.get_allocation_count();
        auto initial_bytes = memory_mgr.get_allocated_bytes();
        
        constexpr size_t alloc_size = 1024;
        auto ptr = memory_mgr.allocate(alloc_size);
        
        REQUIRE(memory_mgr.get_allocation_count() == initial_count + 1);
        REQUIRE(memory_mgr.get_allocated_bytes() >= initial_bytes + alloc_size);
        
        memory_mgr.deallocate(ptr);
        
        REQUIRE(memory_mgr.get_allocation_count() == initial_count);
        REQUIRE(memory_mgr.get_allocated_bytes() == initial_bytes);
    }
    
    SECTION("Peak memory usage") {
        auto initial_peak = memory_mgr.get_peak_allocated_bytes();
        
        auto ptr1 = memory_mgr.allocate(1024);
        auto ptr2 = memory_mgr.allocate(2048);
        
        auto peak_after_allocs = memory_mgr.get_peak_allocated_bytes();
        REQUIRE(peak_after_allocs >= initial_peak + 3072);
        
        memory_mgr.deallocate(ptr1);
        memory_mgr.deallocate(ptr2);
        
        // Peak should remain at the maximum reached
        REQUIRE(memory_mgr.get_peak_allocated_bytes() == peak_after_allocs);
    }
}