#include <stdexcept>
#include <vector>

#include <sw/kpu/components/compute_fabric.hpp>
#include <sw/kpu/components/scratchpad.hpp>

namespace sw::kpu {

// ComputeFabric implementation
ComputeFabric::ComputeFabric(size_t tile_id, ComputeType type, Size systolic_rows, Size systolic_cols)
    : is_computing(false), compute_start_cycle(0), tile_id(tile_id), compute_type(type) {

    // Initialize systolic array if selected
    if (compute_type == ComputeType::SYSTOLIC_ARRAY) {
        systolic_array = std::make_unique<SystolicArray>(systolic_rows, systolic_cols);
    }
}

ComputeFabric::ComputeFabric(const ComputeFabric& other)
    : is_computing(other.is_computing), compute_start_cycle(other.compute_start_cycle),
      current_op(other.current_op), tile_id(other.tile_id), compute_type(other.compute_type) {

    // Deep copy systolic array if it exists
    if (other.systolic_array) {
        systolic_array = std::make_unique<SystolicArray>(*other.systolic_array);
    }
}

ComputeFabric& ComputeFabric::operator=(const ComputeFabric& other) {
    if (this != &other) {
        is_computing = other.is_computing;
        compute_start_cycle = other.compute_start_cycle;
        current_op = other.current_op;
        tile_id = other.tile_id;
        compute_type = other.compute_type;

        // Deep copy systolic array if it exists
        if (other.systolic_array) {
            systolic_array = std::make_unique<SystolicArray>(*other.systolic_array);
        } else {
            systolic_array.reset();
        }
    }
    return *this;
}

void ComputeFabric::start_matmul(const MatMulConfig& config) {
    if (is_computing) {
        throw std::runtime_error("ComputeFabric is already busy");
    }

    current_op = config;
    is_computing = true;
    compute_start_cycle = 0; // Will be set by the caller

    // Route to appropriate implementation
    if (compute_type == ComputeType::SYSTOLIC_ARRAY && systolic_array) {
        SystolicArray::MatMulConfig systolic_config;
        systolic_config.m = config.m;
        systolic_config.n = config.n;
        systolic_config.k = config.k;
        systolic_config.a_addr = config.a_addr;
        systolic_config.b_addr = config.b_addr;
        systolic_config.c_addr = config.c_addr;
        systolic_config.scratchpad_id = config.scratchpad_id;
        systolic_config.completion_callback = config.completion_callback;

        systolic_array->start_matmul(systolic_config);
    }
}

bool ComputeFabric::update(Cycle current_cycle, std::vector<Scratchpad>& scratchpads) {
    if (!is_computing) {
        return false;
    }

    if (compute_start_cycle == 0) {
        compute_start_cycle = current_cycle;
    }

    // Route to appropriate implementation
    if (compute_type == ComputeType::SYSTOLIC_ARRAY && systolic_array) {
        bool completed = systolic_array->update(current_cycle, scratchpads);
        if (completed) {
            is_computing = false;
            return true;
        }
        return false;
    } else {
        // Basic matrix multiplication implementation
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
    if (compute_type == ComputeType::SYSTOLIC_ARRAY && systolic_array) {
        return systolic_array->estimate_cycles(m, n, k);
    } else {
        // Simplified model: assume 1 cycle per MAC operation
        return m * n * k;
    }
}

Size ComputeFabric::get_systolic_rows() const {
    if (systolic_array) {
        return systolic_array->get_rows();
    }
    return 0;
}

Size ComputeFabric::get_systolic_cols() const {
    if (systolic_array) {
        return systolic_array->get_cols();
    }
    return 0;
}

void ComputeFabric::reset() {
    is_computing = false;
    compute_start_cycle = 0;

    if (systolic_array) {
        systolic_array->reset();
    }
}

} // namespace sw::kpu