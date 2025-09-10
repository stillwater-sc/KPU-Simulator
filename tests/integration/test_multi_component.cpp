#include <catch2/catch_test_macros.hpp>
#include <sw/kpu/kpu_simulator.hpp>
#include <sw/driver/memory_manager.hpp>

using namespace sw::kpu;

TEST_CASE("Multi-component integration", "[integration][multi_component]") {
    KpuSimulator simulator;
    
    SECTION("Memory and simulator integration") {
        REQUIRE(simulator.initialize());
        
        auto& memory_mgr = simulator.get_memory_manager();
        
        // Test memory allocation through simulator
        void* ptr1 = memory_mgr.allocate(1024);
        REQUIRE(ptr1 != nullptr);
        
        void* ptr2 = memory_mgr.allocate(2048);
        REQUIRE(ptr2 != nullptr);
        REQUIRE(ptr1 != ptr2);
        
        REQUIRE(memory_mgr.get_allocation_count() == 2);
        
        memory_mgr.deallocate(ptr1);
        memory_mgr.deallocate(ptr2);
        
        REQUIRE(memory_mgr.get_allocation_count() == 0);
        
        simulator.shutdown();
    }
}