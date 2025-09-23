#include <stdexcept>
#include <vector>

#include <sw/kpu/components/compute_fabric.hpp>
#include <sw/kpu/components/scratchpad.hpp>

namespace sw::kpu {

// ComputeFabric implementation
ComputeFabric::ComputeFabric(size_t tile_id)
    : is_computing(false), compute_start_cycle(0), tile_id(tile_id) {
}

void ComputeFabric::start_matmul(const MatMulConfig& config) {
    if (is_computing) {
        throw std::runtime_error("ComputeFabric is already busy");
    }

    current_op = config;
    is_computing = true;
    compute_start_cycle = 0; // Will be set by the caller
}

bool ComputeFabric::update(Cycle current_cycle, std::vector<Scratchpad>& scratchpads) {
    if (!is_computing) {
        return false;
    }

    if (compute_start_cycle == 0) {
        compute_start_cycle = current_cycle;
    }

    Cycle required_cycles = estimate_cycles(current_op.m, current_op.n, current_op.k);

    if (current_cycle - compute_start_cycle >= required_cycles) {
        // Operation completed
        execute_matmul(scratchpads);

        if (current_op.completion_callback) {
            current_op.completion_callback();
        }

        is_computing = false;
        return true;
    }

    return false;
}

void ComputeFabric::execute_matmul(std::vector<Scratchpad>& scratchpads) {
    if (current_op.scratchpad_id >= scratchpads.size()) {
        throw std::out_of_range("Invalid scratchpad ID for matmul operation");
    }

    auto& scratchpad = scratchpads[current_op.scratchpad_id];

    // Read matrices from scratchpad
    Size a_size = current_op.m * current_op.k * sizeof(float);
    Size b_size = current_op.k * current_op.n * sizeof(float);
    Size c_size = current_op.m * current_op.n * sizeof(float);

    std::vector<float> a(current_op.m * current_op.k);
    std::vector<float> b(current_op.k * current_op.n);
    std::vector<float> c(current_op.m * current_op.n, 0.0f);

    scratchpad.read(current_op.a_addr, a.data(), a_size);
    scratchpad.read(current_op.b_addr, b.data(), b_size);

    // Perform matrix multiplication: C = A * B
    for (Size i = 0; i < current_op.m; ++i) {
        for (Size j = 0; j < current_op.n; ++j) {
            float sum = 0.0f;
            for (Size k = 0; k < current_op.k; ++k) {
                sum += a[i * current_op.k + k] * b[k * current_op.n + j];
            }
            c[i * current_op.n + j] = sum;
        }
    }

    // Write result back to scratchpad
    scratchpad.write(current_op.c_addr, c.data(), c_size);
}

Cycle ComputeFabric::estimate_cycles(Size m, Size n, Size k) const {
    // Simplified model: assume 1 cycle per MAC operation
    return m * n * k;
}

void ComputeFabric::reset() {
    is_computing = false;
    compute_start_cycle = 0;
}

} // namespace sw::kpu