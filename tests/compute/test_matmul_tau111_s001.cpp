#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstring>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/compute/matmul_tau111_s001.hpp>

using namespace sw::kpu;

// Test fixture for Compute engine tests
class MatmulTau111S001TestFixture {
public:
    using SA = sw::compute::MatmulTau111S001<float, float, float>;
    std::unique_ptr< SA > sut;

    MatmulTau111S001TestFixture() {
        // Basic configuration for the simulator
        sut = std::make_unique<SA>();
    }

    // Helper to generate test matrix data
    std::vector<float> generate_matrix(size_t rows, size_t cols, float start_value = 1.0f) {
        std::vector<float> matrix(rows * cols);
        for (size_t i = 0; i < rows * cols; ++i) {
            matrix[i] = start_value + static_cast<float>(i);
        }
        return matrix;
    }

    std::vector<float> matmul(const std::vector<float>& a, const std::vector<float>& b) {
        size_t m = 4, n = 4, k = 4; // Fixed sizes for this example
        std::vector<float> c(m * n, 0.0f);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t ki = 0; ki < k; ++ki) {
                    c[i * n + j] += a[i * k + ki] * b[ki * n + j];
                }
            }
        }
        return c;
	}

    // Helper to verify matrix multiplication result
    bool verify_matmul (const std::vector<float>& a, const std::vector<float>& b,
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

TEST_CASE_METHOD(MatmulTau111S001TestFixture, "Matmul Basic Functionality", "[compute][basic]") {
    SECTION("Matmul") {
        // Test parameters
		auto A = generate_matrix(4, 8, 1.0f);  // [[1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16], ...]
		auto B = generate_matrix(8, 4, 1.0f);  // [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], ...]
        auto C = matmul(A, B);

#ifdef LATER
        // Load inputs into PE
        sut->reset();  // this will reset the accumulator as well
        constexpr unsigned VAR_A = 0;
        constexpr unsigned VAR_B = 1;
        constexpr unsigned VAR_C = 2; // Accumulator
        sut->load_input(VAR_A, a);
        sut->load_input(VAR_B, b);
        sut->start(); // Start processing
        // Run simulation until compute completes
        while (!sut->is_busy()) {
            sut->cycle();
        }
        // run till complete
        REQUIRE_FALSE(sut->is_busy());

        // Get and Verify result
        float c = sut->result();

        REQUIRE(verify_matmul(expected_result, c));
#endif
    }
}
