#include <catch2/catch_test_macros.hpp>
#include "sw/system/toplevel.hpp"

using namespace sw::sim;

TEST_CASE("TopLevelSimulator construction", "[system][toplevel]") {
    SECTION("can be constructed") {
        TopLevelSimulator simulator;
        REQUIRE_FALSE(simulator.is_initialized());
    }
}

TEST_CASE("TopLevelSimulator initialization lifecycle", "[system][toplevel]") {
    TopLevelSimulator simulator;
    
    SECTION("starts uninitialized") {
        REQUIRE_FALSE(simulator.is_initialized());
    }
    
    SECTION("can be initialized") {
        REQUIRE(simulator.initialize());
        REQUIRE(simulator.is_initialized());
    }
    
    SECTION("initialize is idempotent") {
        REQUIRE(simulator.initialize());
        REQUIRE(simulator.is_initialized());
        
        // Second call should still succeed
        REQUIRE(simulator.initialize());
        REQUIRE(simulator.is_initialized());
    }
    
    SECTION("can be shut down after initialization") {
        REQUIRE(simulator.initialize());
        REQUIRE(simulator.is_initialized());
        
        simulator.shutdown();
        REQUIRE_FALSE(simulator.is_initialized());
    }
    
    SECTION("shutdown is idempotent") {
        REQUIRE(simulator.initialize());
        simulator.shutdown();
        REQUIRE_FALSE(simulator.is_initialized());
        
        // Second shutdown should not crash
        simulator.shutdown();
        REQUIRE_FALSE(simulator.is_initialized());
    }
}

TEST_CASE("TopLevelSimulator self test", "[system][toplevel]") {
    TopLevelSimulator simulator;
    
    SECTION("self test fails when not initialized") {
        REQUIRE_FALSE(simulator.run_self_test());
    }
    
    SECTION("self test passes when initialized") {
        REQUIRE(simulator.initialize());
        REQUIRE(simulator.run_self_test());
    }
    
    SECTION("self test fails after shutdown") {
        REQUIRE(simulator.initialize());
        REQUIRE(simulator.run_self_test());
        
        simulator.shutdown();
        REQUIRE_FALSE(simulator.run_self_test());
    }
}

TEST_CASE("TopLevelSimulator full lifecycle", "[system][toplevel]") {
    TopLevelSimulator simulator;
    
    // Complete lifecycle test
    REQUIRE_FALSE(simulator.is_initialized());
    
    REQUIRE(simulator.initialize());
    REQUIRE(simulator.is_initialized());
    REQUIRE(simulator.run_self_test());
    
    simulator.shutdown();
    REQUIRE_FALSE(simulator.is_initialized());
    REQUIRE_FALSE(simulator.run_self_test());
    
    // Can be reinitialized
    REQUIRE(simulator.initialize());
    REQUIRE(simulator.is_initialized());
    REQUIRE(simulator.run_self_test());
    
    simulator.shutdown();
    REQUIRE_FALSE(simulator.is_initialized());
}