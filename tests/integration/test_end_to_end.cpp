#include <catch2/catch_test_macros.hpp>
#include <sw/system/toplevel.hpp>

using namespace sw::sim;

TEST_CASE("End-to-end simulator test", "[integration][end_to_end]") {
    SystemSimulator simulator;
    
    SECTION("Complete workflow") {
        REQUIRE(simulator.initialize());
        REQUIRE(simulator.is_initialized());
        REQUIRE(simulator.run_self_test());
        simulator.shutdown();
        REQUIRE_FALSE(simulator.is_initialized());
    }
}