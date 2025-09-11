#include <catch2/catch_test_macros.hpp>
#include <sw/system/toplevel.hpp>
#include <sw/driver/memory_manager.hpp>

using namespace sw::sim;

TEST_CASE("Multi-component integration", "[integration][multi_component]") {
    TopLevelSimulator simulator;
    
    SECTION("Memory and simulator integration") {
        REQUIRE(simulator.initialize());
        
        //auto& memory_mgr = simulator.get_memory_manager();
		sw::driver::MemoryManager memory_mgr;

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