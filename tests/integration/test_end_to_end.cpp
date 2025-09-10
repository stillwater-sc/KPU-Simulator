#include <catch2/catch_test_macros.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

TEST_CASE("End-to-end simulator test", "[integration][end_to_end]") {
    KpuSimulator simulator;
    
    SECTION("Complete workflow") {
        REQUIRE(simulator.initialize());
        REQUIRE(simulator.is_initialized());
        REQUIRE(simulator.run_self_test());
        simulator.shutdown();
        REQUIRE_FALSE(simulator.is_initialized());
    }
}