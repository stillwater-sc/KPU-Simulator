#pragma once

#include <memory>
#include <vector>
#include <queue>
#include <cstdint>
#include <functional>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251) // DLL interface warnings
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

#include <sw/concepts.hpp>
#include <sw/kpu/components/scratchpad.hpp>

namespace sw::kpu {

// Forward declarations
class Streamer;

// Processing Element (PE) for systolic array
class KPU_API ProcessingElement {
public:
    ProcessingElement(size_t row, size_t col);

    // Data inputs
    void set_a_input(float value) { a_input = value; }
    void set_b_input(float value) { b_input = value; }

    // Data outputs (for propagation)
    float get_a_output() const { return a_output; }
    float get_b_output() const { return b_output; }
    float get_c_output() const { return c_accumulator; }

    // Process one cycle
    void cycle();

    // Reset PE state
    void reset();

    // Configuration
    size_t get_row() const { return row_id; }
    size_t get_col() const { return col_id; }

private:
    size_t row_id, col_id;

    // Data registers
    float a_input, a_output;
    float b_input, b_output;
    float c_accumulator;

    // Control state
    bool accumulating;
    Cycle last_valid_cycle;
};

// Systolic Array for matrix multiplication using output-stationary schedule
class KPU_API SystolicArray {
public:
    static constexpr Size DEFAULT_ROWS = 16;
    static constexpr Size DEFAULT_COLS = 16;

    struct MatMulConfig {
        Size m, n, k; // Matrix dimensions: C[m,n] = A[m,k] * B[k,n]
        Address a_addr, b_addr, c_addr; // Addresses in scratchpad
        size_t scratchpad_id; // Which scratchpad to use
        std::function<void()> completion_callback;
    };

    // Bus directions for data flow
    enum class BusDirection {
        HORIZONTAL,  // A matrix data flows horizontally (left to right)
        VERTICAL,    // B matrix data flows vertically (top to bottom)
        DIAGONAL     // C matrix data flows diagonally (for evacuation)
    };

private:
    // Array configuration
    Size num_rows, num_cols;

    // Processing elements
    std::vector<std::vector<std::unique_ptr<ProcessingElement>>> pe_array;

    // Data buses for systolic flow
    std::vector<std::queue<float>> horizontal_bus; // A data (one per row)
    std::vector<std::queue<float>> vertical_bus;   // B data (one per column)
    std::vector<std::queue<float>> diagonal_bus;   // C data evacuation

    // Operation state
    bool is_computing;
    Cycle compute_start_cycle;
    MatMulConfig current_op;

    // Streaming state
    Size current_a_row, current_a_col;
    Size current_b_row, current_b_col;
    Size current_c_row, current_c_col;
    Size cycles_completed;

    // Staggering for systolic timing
    std::vector<Cycle> row_start_cycles;
    std::vector<Cycle> col_start_cycles;

public:
    explicit SystolicArray(Size rows = DEFAULT_ROWS, Size cols = DEFAULT_COLS);
    ~SystolicArray() = default;

    // Custom copy and move semantics for std::vector compatibility
    SystolicArray(const SystolicArray& other);
    SystolicArray& operator=(const SystolicArray& other);
    SystolicArray(SystolicArray&&) = default;
    SystolicArray& operator=(SystolicArray&&) = default;

    // Matrix multiplication operations
    void start_matmul(const MatMulConfig& config);
    bool update(Cycle current_cycle, std::vector<Scratchpad>& scratchpads);
    bool is_busy() const { return is_computing; }
    void reset();

    // Configuration
    Size get_rows() const { return num_rows; }
    Size get_cols() const { return num_cols; }
    Size get_total_pes() const { return num_rows * num_cols; }

    // Streaming interface for integration with Streamer components
    void stream_a_data(const std::vector<float>& data, Size row_offset);
    void stream_b_data(const std::vector<float>& data, Size col_offset);
    std::vector<float> evacuate_c_data(Size max_elements);

    // Performance metrics
    Cycle estimate_cycles(Size m, Size n, Size k) const;
    Size calculate_throughput() const;

private:
    // Internal processing
    void cycle_pe_array(Cycle current_cycle);
    void load_a_data(Cycle current_cycle, std::vector<Scratchpad>& scratchpads);
    void load_b_data(Cycle current_cycle, std::vector<Scratchpad>& scratchpads);
    void evacuate_c_data(Cycle current_cycle, std::vector<Scratchpad>& scratchpads);

    // Bus management
    void propagate_horizontal_bus();
    void propagate_vertical_bus();
    void propagate_diagonal_bus();

    // Timing and staggering
    bool should_start_row(Size row, Cycle current_cycle) const;
    bool should_start_col(Size col, Cycle current_cycle) const;
    Size calculate_stagger_delay(Size position) const;

    // Data loading helpers
    void load_matrix_a_tile(const std::vector<float>& matrix_a, Size tile_row, Size tile_col);
    void load_matrix_b_tile(const std::vector<float>& matrix_b, Size tile_row, Size tile_col);
    void store_matrix_c_tile(std::vector<float>& matrix_c, Size tile_row, Size tile_col);

    // Address calculation
    Address calculate_matrix_address(Address base_addr, Size row, Size col, Size width, Size element_size) const;

    // Temporary simple implementation for testing
    void perform_direct_matrix_multiply(std::vector<Scratchpad>& scratchpads);
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif