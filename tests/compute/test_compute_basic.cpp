#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/compute/dot_pe.hpp>

using namespace sw::kpu;

// Test fixture for Compute engine tests
class ComputeTestFixture {
public:
    using DotPE = sw::compute::DotProductAccumulator<float, float, float>;
    std::unique_ptr< DotPE > sut;

    ComputeTestFixture() {
        // Basic configuration for the simulator
        sut = std::make_unique<DotPE>();
    }

    // Helper to generate test matrix data

    // Helper to verify compute result
    bool verify_compute(float expected, float actual, float tolerance = 1e-5f) {
        return std::abs(expected - actual) <= tolerance;
    }

};

TEST_CASE_METHOD(ComputeTestFixture, "Compute Basic Functionality", "[compute][basic]") {
    SECTION("Single Dot Product PE works correctly") {
        // Test parameters
        float a = 3.0f;
        float b = 4.0f;
        float expected_result = a * b;

        // Load inputs into PE
        sut->reset();  // this will reset the accumulator as well
        constexpr unsigned VAR_A = 0;
        constexpr unsigned VAR_B = 1;
        constexpr unsigned VAR_C = 2; // Accumulator
        sut->load_input(VAR_A, a);
        sut->load_input(VAR_B, b);
        sut->start(); // Start processing
        // Run simulation until compute completes
        while (sut->is_busy()) {
            sut->cycle();
        }
        // run till complete
        REQUIRE_FALSE(sut->is_busy());

        // Get and Verify result
        float c = sut->result();

        REQUIRE(verify_compute(expected_result, c));
    }
}
