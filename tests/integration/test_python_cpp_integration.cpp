#include <catch2/catch_test_macros.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

TEST_CASE("Python-C++ integration", "[integration][python_cpp]") {
    KpuSimulator simulator;
    
    SECTION("Basic integration test") {
        // This test verifies that the C++ simulator works
        // Python integration would be tested separately
        REQUIRE(simulator.initialize());
        REQUIRE(simulator.is_initialized());
        
        // Test memory manager access
        auto& memory_mgr = simulator.get_memory_manager();
        void* ptr = memory_mgr.allocate(512);
        REQUIRE(ptr != nullptr);
        memory_mgr.deallocate(ptr);
        
        simulator.shutdown();
    }
}