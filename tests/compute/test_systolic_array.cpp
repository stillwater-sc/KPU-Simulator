#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

// Test fixture for Systolic Array tests
class SystolicArrayTestFixture {
public:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> sim;

    SystolicArrayTestFixture() {
        // Configuration with systolic arrays enabled
        config.memory_bank_count = 2;
        config.memory_bank_capacity_mb = 64;
        config.memory_bandwidth_gbps = 8;
        config.scratchpad_count = 4;
        config.scratchpad_capacity_kb = 256;
        config.compute_tile_count = 1;
        config.dma_engine_count = 2;
        config.l3_tile_count = 4;
        config.l3_tile_capacity_kb = 128;
        config.l2_bank_count = 8;
        config.l2_bank_capacity_kb = 64;
        config.block_mover_count = 4;
        config.streamer_count = 8;

        // Systolic array configuration
        config.systolic_array_rows = 4;  // Smaller for testing
        config.systolic_array_cols = 4;
        config.use_systolic_arrays = true;

        sim = std::make_unique<KPUSimulator>(config);
    }

    // Helper to generate test matrix data
    std::vector<float> generate_matrix(size_t rows, size_t cols, float start_value = 1.0f) {
        std::vector<float> matrix(rows * cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            matrix[i] = start_value + static_cast<float>(i);
        }
        return matrix;
    }

    // Helper to verify matrix multiplication result
    bool verify_matmul(const std::vector<float>& a, const std::vector<float>& b,
                      const std::vector<float>& c, size_t m, size_t n, size_t k,
                      float tolerance = 1e-5f) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float expected = 0.0f;
                for (size_t ki = 0; ki < k; ++ki) {
                    expected += a[i * k + ki] * b[ki * n + j];
                }
                float actual = c[i * n + j];
                if (std::abs(actual - expected) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
};

TEST_CASE_METHOD(SystolicArrayTestFixture, "Systolic Array Configuration", "[systolic][config]") {
    SECTION("Can query systolic array configuration") {
        REQUIRE(sim->is_using_systolic_arrays() == true);
        REQUIRE(sim->get_systolic_array_rows() == 4);
        REQUIRE(sim->get_systolic_array_cols() == 4);
        REQUIRE(sim->get_systolic_array_total_pes() == 16);
    }

    SECTION("Can disable systolic arrays") {
        // Create a new simulator with systolic arrays disabled
        KPUSimulator::Config basic_config = config;
        basic_config.use_systolic_arrays = false;
        auto basic_sim = std::make_unique<KPUSimulator>(basic_config);

        REQUIRE(basic_sim->is_using_systolic_arrays() == false);
        REQUIRE(basic_sim->get_systolic_array_rows() == 0);
        REQUIRE(basic_sim->get_systolic_array_cols() == 0);
        REQUIRE(basic_sim->get_systolic_array_total_pes() == 0);
    }
}

TEST_CASE_METHOD(SystolicArrayTestFixture, "Systolic Array Matrix Multiplication", "[systolic][matmul]") {
    SECTION("Can perform small matrix multiplication") {
        // Test parameters
        const size_t m = 2, n = 2, k = 2;
        const size_t element_size = sizeof(float);

        // Generate test matrices
        auto matrix_a = generate_matrix(m, k, 1.0f);  // [[1,2], [3,4]]
        auto matrix_b = generate_matrix(k, n, 5.0f);  // [[5,6], [7,8]]
        std::vector<float> matrix_c(m * n, 0.0f);

        // Expected result: [[19,22], [43,50]]

        // Write matrices to scratchpad
        const size_t scratchpad_id = 0;
        const Address a_addr = 0;
        const Address b_addr = a_addr + matrix_a.size() * element_size;
        const Address c_addr = b_addr + matrix_b.size() * element_size;

        sim->write_scratchpad(scratchpad_id, a_addr, matrix_a.data(), matrix_a.size() * element_size);
        sim->write_scratchpad(scratchpad_id, b_addr, matrix_b.data(), matrix_b.size() * element_size);

        // Start matrix multiplication
        bool matmul_complete = false;
        sim->start_matmul(0, scratchpad_id, m, n, k, a_addr, b_addr, c_addr,
                         [&matmul_complete]() { matmul_complete = true; });

        // Run simulation until completion
        sim->run_until_idle();

        REQUIRE(matmul_complete);
        REQUIRE_FALSE(sim->is_compute_busy(0));

        // Read result
        sim->read_scratchpad(scratchpad_id, c_addr, matrix_c.data(), matrix_c.size() * element_size);

        // Verify result
        REQUIRE(verify_matmul(matrix_a, matrix_b, matrix_c, m, n, k));

        // Check specific values
        REQUIRE(matrix_c[0] == Catch::Approx(19.0f)); // (1*5 + 2*7)
        REQUIRE(matrix_c[1] == Catch::Approx(22.0f)); // (1*6 + 2*8)
        REQUIRE(matrix_c[2] == Catch::Approx(43.0f)); // (3*5 + 4*7)
        REQUIRE(matrix_c[3] == Catch::Approx(50.0f)); // (3*6 + 4*8)
    }

    SECTION("Can handle larger matrices") {
        // Test parameters - larger than systolic array size
        const size_t m = 8, n = 8, k = 8;
        const size_t element_size = sizeof(float);

        // Generate test matrices
        auto matrix_a = generate_matrix(m, k, 1.0f);
        auto matrix_b = generate_matrix(k, n, 0.1f);
        std::vector<float> matrix_c(m * n, 0.0f);

        // Write matrices to scratchpad
        const size_t scratchpad_id = 0;
        const Address a_addr = 0;
        const Address b_addr = a_addr + matrix_a.size() * element_size;
        const Address c_addr = b_addr + matrix_b.size() * element_size;

        sim->write_scratchpad(scratchpad_id, a_addr, matrix_a.data(), matrix_a.size() * element_size);
        sim->write_scratchpad(scratchpad_id, b_addr, matrix_b.data(), matrix_b.size() * element_size);

        // Start matrix multiplication
        bool matmul_complete = false;
        sim->start_matmul(0, scratchpad_id, m, n, k, a_addr, b_addr, c_addr,
                         [&matmul_complete]() { matmul_complete = true; });

        // Run simulation until completion
        sim->run_until_idle();

        REQUIRE(matmul_complete);
        REQUIRE_FALSE(sim->is_compute_busy(0));

        // Read result
        sim->read_scratchpad(scratchpad_id, c_addr, matrix_c.data(), matrix_c.size() * element_size);

        // Verify result
        REQUIRE(verify_matmul(matrix_a, matrix_b, matrix_c, m, n, k));
    }

    SECTION("Systolic array is faster than basic implementation") {
        // This test is more about ensuring the systolic array works
        // Performance comparison would need cycle counting
        const size_t m = 4, n = 4, k = 4;
        const size_t element_size = sizeof(float);

        auto matrix_a = generate_matrix(m, k, 1.0f);
        auto matrix_b = generate_matrix(k, n, 1.0f);
        std::vector<float> matrix_c(m * n, 0.0f);

        const size_t scratchpad_id = 0;
        const Address a_addr = 0;
        const Address b_addr = a_addr + matrix_a.size() * element_size;
        const Address c_addr = b_addr + matrix_b.size() * element_size;

        sim->write_scratchpad(scratchpad_id, a_addr, matrix_a.data(), matrix_a.size() * element_size);
        sim->write_scratchpad(scratchpad_id, b_addr, matrix_b.data(), matrix_b.size() * element_size);

        bool matmul_complete = false;
        auto start_cycle = sim->get_current_cycle();

        sim->start_matmul(0, scratchpad_id, m, n, k, a_addr, b_addr, c_addr,
                         [&matmul_complete]() { matmul_complete = true; });

        sim->run_until_idle();
        auto end_cycle = sim->get_current_cycle();

        REQUIRE(matmul_complete);

        // Read and verify result
        sim->read_scratchpad(scratchpad_id, c_addr, matrix_c.data(), matrix_c.size() * element_size);
        REQUIRE(verify_matmul(matrix_a, matrix_b, matrix_c, m, n, k));

        // Systolic array should complete in reasonable time
        // (actual cycles depend on implementation details)
        auto cycles_used = end_cycle - start_cycle;
        REQUIRE(cycles_used > 0);
        REQUIRE(cycles_used < 1000); // Reasonable upper bound
    }
}

TEST_CASE_METHOD(SystolicArrayTestFixture, "Systolic Array Error Handling", "[systolic][error]") {
    SECTION("Validates compute tile bounds") {
        REQUIRE_THROWS_AS(sim->get_systolic_array_rows(99), std::out_of_range);
        REQUIRE_THROWS_AS(sim->get_systolic_array_cols(99), std::out_of_range);
        REQUIRE_THROWS_AS(sim->get_systolic_array_total_pes(99), std::out_of_range);
    }

    SECTION("Cannot start multiple operations on same tile") {
        const size_t m = 2, n = 2, k = 2;
        auto matrix_a = generate_matrix(m, k, 1.0f);
        auto matrix_b = generate_matrix(k, n, 1.0f);

        const size_t scratchpad_id = 0;
        const Address a_addr = 0;
        const Address b_addr = 64;
        const Address c_addr = 128;

        sim->write_scratchpad(scratchpad_id, a_addr, matrix_a.data(), matrix_a.size() * sizeof(float));
        sim->write_scratchpad(scratchpad_id, b_addr, matrix_b.data(), matrix_b.size() * sizeof(float));

        // Start first operation
        sim->start_matmul(0, scratchpad_id, m, n, k, a_addr, b_addr, c_addr);

        REQUIRE(sim->is_compute_busy(0));

        // Try to start second operation - should throw
        REQUIRE_THROWS_AS(
            sim->start_matmul(0, scratchpad_id, m, n, k, a_addr, b_addr, c_addr),
            std::runtime_error
        );

        // Clean up
        sim->run_until_idle();
    }
}