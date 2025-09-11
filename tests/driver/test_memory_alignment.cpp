#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <sw/driver/memory_manager.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

using namespace sw::driver;

// Helper function to check if a pointer is aligned to a specific type's alignment
template<typename T>
bool is_aligned(void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % alignof(T) == 0;
}

// Helper function to check if a pointer is aligned to a given boundary
bool is_aligned(void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// Helper function to get the alignment of a pointer
uintptr_t get_alignment(void* ptr) {
    auto addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr == 0) return SIZE_MAX;
    
    uintptr_t alignment = 1;
    while ((addr & alignment) == 0 && alignment <= 4096) {
        alignment <<= 1;
    }
    return alignment >> 1;
}

#ifdef DRIVER_TESTS

TEST_CASE("Default memory alignment", "[memory][alignment]") {
    MemoryManager memory_mgr;
    
    SECTION("Standard scalar types") {
        auto ptr_char = memory_mgr.allocate(sizeof(char));
        auto ptr_int = memory_mgr.allocate(sizeof(int));
        auto ptr_double = memory_mgr.allocate(sizeof(double));
        auto ptr_long_long = memory_mgr.allocate(sizeof(long long));
        
        REQUIRE(is_aligned<char>(ptr_char));
        REQUIRE(is_aligned<int>(ptr_int));
        REQUIRE(is_aligned<double>(ptr_double));
        REQUIRE(is_aligned<long long>(ptr_long_long));
        
        memory_mgr.deallocate(ptr_char);
        memory_mgr.deallocate(ptr_int);
        memory_mgr.deallocate(ptr_double);
        memory_mgr.deallocate(ptr_long_long);
    }
    
    SECTION("Default alignment guarantee") {
        // Most allocators guarantee at least alignof(std::max_align_t)
        constexpr size_t default_alignment = alignof(std::max_align_t);
        
        for (size_t size : {1, 7, 15, 31, 63, 127, 255}) {
            auto ptr = memory_mgr.allocate(size);
            REQUIRE(ptr != nullptr);
            REQUIRE(get_alignment(ptr) >= default_alignment);
            memory_mgr.deallocate(ptr);
        }
    }
}

TEST_CASE("Explicit memory alignment", "[memory][alignment][aligned]") {
    AlignedMemoryManager memory_mgr;
    
    SECTION("Power-of-two alignments") {
        size_t alignment = GENERATE(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024);
        constexpr size_t size = 1024;
        
        auto ptr = memory_mgr.allocate_aligned(size, alignment);
        REQUIRE(ptr != nullptr);
        REQUIRE(is_aligned(ptr, alignment));
        REQUIRE(get_alignment(ptr) >= alignment);
        
        memory_mgr.deallocate_aligned(ptr);
    }
    
    SECTION("Cache line alignment") {
        constexpr size_t cache_line_size = 64;  // Common cache line size
        constexpr size_t size = 1024;
        
        auto ptr = memory_mgr.allocate_aligned(size, cache_line_size);
        REQUIRE(ptr != nullptr);
        REQUIRE(is_aligned(ptr, cache_line_size));
        
        // Verify we can write to the memory without issues
        std::memset(ptr, 0xAA, size);
        
        memory_mgr.deallocate_aligned(ptr);
    }
    
    SECTION("SIMD alignment requirements") {
        // Test various SIMD alignment requirements
        struct AlignmentTest {
            size_t alignment;
            const char* description;
        };
        
        std::vector<AlignmentTest> tests = {
            {16, "SSE (128-bit)"},
            {32, "AVX (256-bit)"},
            {64, "AVX-512 (512-bit)"}
        };
        
        for (const auto& test : tests) {
            INFO("Testing " << test.description << " alignment");
            
            constexpr size_t size = 1024;
            auto ptr = memory_mgr.allocate_aligned(size, test.alignment);
            
            REQUIRE(ptr != nullptr);
            REQUIRE(is_aligned(ptr, test.alignment));
            
            // Test actual SIMD-like access pattern
            auto typed_ptr = static_cast<uint64_t*>(ptr);
            for (size_t i = 0; i < size / sizeof(uint64_t); ++i) {
                typed_ptr[i] = i;
            }
            
            memory_mgr.deallocate_aligned(ptr);
        }
    }
}

TEST_CASE("Memory alignment edge cases", "[memory][alignment][edge]") {
    AlignedMemoryManager memory_mgr;
    
    SECTION("Large alignment") {
        constexpr size_t large_alignment = 4096;  // Page alignment
        constexpr size_t size = 8192;
        
        auto ptr = memory_mgr.allocate_aligned(size, large_alignment);
        if (ptr != nullptr) {  // May fail on some systems
            REQUIRE(is_aligned(ptr, large_alignment));
            memory_mgr.deallocate_aligned(ptr);
        }
    }
    
    SECTION("Alignment larger than size") {
        constexpr size_t small_size = 16;
        constexpr size_t large_alignment = 256;
        
        auto ptr = memory_mgr.allocate_aligned(small_size, large_alignment);
        REQUIRE(ptr != nullptr);
        REQUIRE(is_aligned(ptr, large_alignment));
        
        memory_mgr.deallocate_aligned(ptr);
    }
    
    SECTION("Non-power-of-two alignment") {
        // Some alignments that are not powers of 2
        for (size_t alignment : {3, 6, 12, 24, 48}) {
            constexpr size_t size = 1024;
            
            // This might throw or return nullptr depending on implementation
            try {
                auto ptr = memory_mgr.allocate_aligned(size, alignment);
                if (ptr != nullptr) {
                    REQUIRE(is_aligned(ptr, alignment));
                    memory_mgr.deallocate_aligned(ptr);
                }
            } catch (const std::invalid_argument&) {
                // Non-power-of-2 alignments may not be supported
                SUCCEED("Non-power-of-2 alignment rejected as expected");
            }
        }
    }
    
    SECTION("Zero alignment") {
        constexpr size_t size = 1024;
        
        REQUIRE_THROWS_AS(
            memory_mgr.allocate_aligned(size, 0),
            std::invalid_argument
        );
    }
}

TEST_CASE("Alignment with different sizes", "[memory][alignment][sizes]") {
    AlignedMemoryManager memory_mgr;
    constexpr size_t alignment = 64;
    
    SECTION("Various allocation sizes") {
        auto size = GENERATE(1, 17, 33, 65, 129, 257, 513, 1025, 2049);
        
        auto ptr = memory_mgr.allocate_aligned(size, alignment);
        REQUIRE(ptr != nullptr);
        REQUIRE(is_aligned(ptr, alignment));
        
        // Verify we can use the full requested size
        std::memset(ptr, 0x55, size);
        
        memory_mgr.deallocate_aligned(ptr);
    }
    
    SECTION("Size equals alignment") {
        auto ptr = memory_mgr.allocate_aligned(alignment, alignment);
        REQUIRE(ptr != nullptr);
        REQUIRE(is_aligned(ptr, alignment));
        
        memory_mgr.deallocate_aligned(ptr);
    }
    
    SECTION("Size is multiple of alignment") {
        constexpr size_t size = alignment * 4;
        
        auto ptr = memory_mgr.allocate_aligned(size, alignment);
        REQUIRE(ptr != nullptr);
        REQUIRE(is_aligned(ptr, alignment));
        
        memory_mgr.deallocate_aligned(ptr);
    }
}

TEST_CASE("Multiple aligned allocations", "[memory][alignment][multiple]") {
    AlignedMemoryManager memory_mgr;
    
    SECTION("Same alignment, different sizes") {
        constexpr size_t alignment = 32;
        std::vector<std::pair<void*, size_t>> allocations;
        
        for (size_t size : {64, 128, 256, 512, 1024}) {
            auto ptr = memory_mgr.allocate_aligned(size, alignment);
            REQUIRE(ptr != nullptr);
            REQUIRE(is_aligned(ptr, alignment));
            allocations.emplace_back(ptr, size);
        }
        
        // Verify all allocations are distinct
        for (size_t i = 0; i < allocations.size(); ++i) {
            for (size_t j = i + 1; j < allocations.size(); ++j) {
                auto [ptr1, size1] = allocations[i];
                auto [ptr2, size2] = allocations[j];
                
                // Check for overlap
                auto addr1 = reinterpret_cast<uintptr_t>(ptr1);
                auto addr2 = reinterpret_cast<uintptr_t>(ptr2);
                
                REQUIRE(((addr1 + size1 <= addr2) || (addr2 + size2 <= addr1)));
            }
        }
        
        // Clean up
        for (auto [ptr, size] : allocations) {
            memory_mgr.deallocate_aligned(ptr);
        }
    }
    
    SECTION("Different alignments") {
        std::vector<std::pair<void*, size_t>> allocations;
        
        for (size_t alignment : {8, 16, 32, 64, 128}) {
            constexpr size_t size = 256;
            auto ptr = memory_mgr.allocate_aligned(size, alignment);
            REQUIRE(ptr != nullptr);
            REQUIRE(is_aligned(ptr, alignment));
            allocations.emplace_back(ptr, alignment);
        }
        
        // Clean up
        for (auto [ptr, alignment] : allocations) {
            memory_mgr.deallocate_aligned(ptr);
        }
    }
}

TEST_CASE("Aligned allocation performance", "[memory][alignment][performance]") {
    AlignedMemoryManager memory_mgr;
    constexpr size_t num_allocations = 1000;
    constexpr size_t alignment = 64;
    constexpr size_t size = 1024;
    
    SECTION("Allocation speed") {
        std::vector<void*> ptrs;
        ptrs.reserve(num_allocations);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < num_allocations; ++i) {
            auto ptr = memory_mgr.allocate_aligned(size, alignment);
            if (ptr) ptrs.push_back(ptr);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Performance expectation (adjust based on requirements)
        REQUIRE(duration.count() < 50000);  // Should complete in less than 50ms
        
        // Verify all allocations are properly aligned
        for (auto ptr : ptrs) {
            REQUIRE(is_aligned(ptr, alignment));
        }
        
        // Clean up
        for (auto ptr : ptrs) {
            memory_mgr.deallocate_aligned(ptr);
        }
    }
} 

#endif