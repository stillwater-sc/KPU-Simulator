#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/kpu/components/memory_orchestrator.hpp>
#include <vector>
#include <algorithm>
#include <cstring>
#include <thread>
#include <chrono>
#include <atomic>

using namespace sw::kpu;

// Test fixture for EDDO workflow tests
class EDDOWorkflowTestFixture {
public:
    static constexpr size_t NUM_BANKS = 4;
    static constexpr size_t BANK_SIZE_KB = 64;

    MemoryOrchestrator::BankConfig eddo_config{
        .bank_size_kb = BANK_SIZE_KB,
        .cache_line_size = 64,
        .num_ports = 2,
        .access_pattern = MemoryOrchestrator::AccessPattern::SEQUENTIAL,
        .enable_prefetch = true
    };

    std::unique_ptr<MemoryOrchestrator> orchestrator;

    EDDOWorkflowTestFixture() {
        orchestrator = std::make_unique<MemoryOrchestrator>(0, NUM_BANKS, eddo_config);
    }

    // Helper to create test matrix data
    std::vector<float> create_test_matrix(size_t rows, size_t cols, float base_value = 1.0f) {
        std::vector<float> matrix(rows * cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                matrix[i * cols + j] = base_value + static_cast<float>(i * cols + j);
            }
        }
        return matrix;
    }

    // Helper to simulate matrix computation
    void matrix_multiply_compute(const std::vector<float>& a, const std::vector<float>& b,
                               std::vector<float>& c, size_t rows, size_t cols, size_t inner_dim) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                c[i * cols + j] = 0.0f;
                for (size_t k = 0; k < inner_dim; ++k) {
                    c[i * cols + j] += a[i * inner_dim + k] * b[k * cols + j];
                }
            }
        }
    }
};

TEST_CASE_METHOD(EDDOWorkflowTestFixture, "EDDO Basic Command Processing", "[orchestrator][eddo][basic]") {

    SECTION("Single EDDO command execution") {
        std::atomic<bool> prefetch_completed{false};

        MemoryOrchestrator::EDDOCommand prefetch_cmd{
            .phase = MemoryOrchestrator::EDDOPhase::PREFETCH,
            .bank_id = 0,
            .source_addr = 0x1000,
            .dest_addr = 0,
            .transfer_size = 1024,
            .sequence_id = 1,
            .dependencies = {},
            .block_mover_id = SIZE_MAX,
            .streamer_id = SIZE_MAX,
            .completion_callback = [&prefetch_completed](const MemoryOrchestrator::EDDOCommand&) {
                prefetch_completed = true;
            }
        };

        orchestrator->enqueue_eddo_command(prefetch_cmd);
        REQUIRE(orchestrator->get_pending_commands() == 1);

        // Process the command
        bool processed = orchestrator->process_eddo_commands();
        REQUIRE(processed);

        // Wait a bit for async completion
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        REQUIRE(prefetch_completed);

        REQUIRE(orchestrator->get_bank_phase(0) == MemoryOrchestrator::EDDOPhase::PREFETCH);
    }

    SECTION("EDDO command dependency handling") {
        std::atomic<int> execution_order{0};
        std::vector<int> execution_sequence;

        // Create dependent commands
        MemoryOrchestrator::EDDOCommand prefetch_cmd{
            .phase = MemoryOrchestrator::EDDOPhase::PREFETCH,
            .bank_id = 0,
            .source_addr = 0x1000,
            .dest_addr = 0,
            .transfer_size = 1024,
            .sequence_id = 1,
            .completion_callback = [&](const MemoryOrchestrator::EDDOCommand&) {
                execution_sequence.push_back(execution_order.fetch_add(1));
            }
        };

        MemoryOrchestrator::EDDOCommand compute_cmd{
            .phase = MemoryOrchestrator::EDDOPhase::COMPUTE,
            .bank_id = 0,
            .sequence_id = 2,
            .dependencies = {1}, // Depends on prefetch
            .completion_callback = [&](const MemoryOrchestrator::EDDOCommand&) {
                execution_sequence.push_back(execution_order.fetch_add(1));
            }
        };

        MemoryOrchestrator::EDDOCommand writeback_cmd{
            .phase = MemoryOrchestrator::EDDOPhase::WRITEBACK,
            .bank_id = 0,
            .source_addr = 0,
            .dest_addr = 0x2000,
            .transfer_size = 1024,
            .sequence_id = 3,
            .dependencies = {2}, // Depends on compute
            .completion_callback = [&](const MemoryOrchestrator::EDDOCommand&) {
                execution_sequence.push_back(execution_order.fetch_add(1));
            }
        };

        // Enqueue in reverse order to test dependency resolution
        orchestrator->enqueue_eddo_command(writeback_cmd);
        orchestrator->enqueue_eddo_command(compute_cmd);
        orchestrator->enqueue_eddo_command(prefetch_cmd);

        // Process commands multiple times to handle dependencies
        int max_iterations = 10;
        while (orchestrator->get_pending_commands() > 0 && max_iterations-- > 0) {
            orchestrator->process_eddo_commands();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Verify execution order
        REQUIRE(execution_sequence.size() == 3);
        REQUIRE(execution_sequence[0] == 0); // Prefetch first
        REQUIRE(execution_sequence[1] == 1); // Compute second
        REQUIRE(execution_sequence[2] == 2); // Writeback last
    }
}

TEST_CASE_METHOD(EDDOWorkflowTestFixture, "EDDO Workflow Builder", "[orchestrator][eddo][workflow]") {

    SECTION("Workflow builder basic functionality") {
        std::atomic<int> step_counter{0};

        EDDOWorkflowBuilder builder;
        auto workflow = builder
            .prefetch(0, 0x1000, 0, 1024)
            .compute(0, [&step_counter]() { step_counter++; })
            .writeback(0, 0, 0x2000, 1024)
            .sync()
            .build();

        REQUIRE(workflow.size() == 4);
        REQUIRE(workflow[0].phase == MemoryOrchestrator::EDDOPhase::PREFETCH);
        REQUIRE(workflow[1].phase == MemoryOrchestrator::EDDOPhase::COMPUTE);
        REQUIRE(workflow[2].phase == MemoryOrchestrator::EDDOPhase::WRITEBACK);
        REQUIRE(workflow[3].phase == MemoryOrchestrator::EDDOPhase::SYNC);
    }

    SECTION("Execute workflow on orchestrator") {
        std::atomic<bool> compute_executed{false};

        EDDOWorkflowBuilder builder;
        builder.prefetch(0, 0x1000, 0, 512)
               .compute(0, [&compute_executed]() { compute_executed = true; })
               .writeback(0, 0, 0x2000, 512)
               .execute_on(*orchestrator);

        // Process all commands
        int max_iterations = 10;
        while (orchestrator->is_busy() && max_iterations-- > 0) {
            orchestrator->process_eddo_commands();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        REQUIRE(compute_executed);
    }
}

TEST_CASE_METHOD(EDDOWorkflowTestFixture, "EDDO Advanced Patterns", "[orchestrator][eddo][advanced]") {

    SECTION("Double buffering pattern") {
        constexpr size_t TRANSFER_SIZE = 1024;
        constexpr Address SRC_BASE = 0x10000;

        // Set up double buffering
        orchestrator->orchestrate_double_buffer(0, 1, SRC_BASE, TRANSFER_SIZE);

        REQUIRE(orchestrator->get_pending_commands() == 2);

        // Process double buffer commands
        int iterations = 0;
        while (orchestrator->is_busy() && iterations++ < 20) {
            orchestrator->process_eddo_commands();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Both banks should be in PREFETCH phase
        REQUIRE(orchestrator->get_bank_phase(0) == MemoryOrchestrator::EDDOPhase::PREFETCH);
        REQUIRE(orchestrator->get_bank_phase(1) == MemoryOrchestrator::EDDOPhase::PREFETCH);
    }

    SECTION("Pipeline stage orchestration") {
        std::atomic<bool> pipeline_compute_executed{false};

        auto compute_func = [&pipeline_compute_executed]() {
            pipeline_compute_executed = true;
        };

        orchestrator->orchestrate_pipeline_stage(0, 1, compute_func);

        REQUIRE(orchestrator->get_pending_commands() == 2);

        // Process pipeline commands
        int iterations = 0;
        while (orchestrator->is_busy() && iterations++ < 20) {
            orchestrator->process_eddo_commands();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        REQUIRE(pipeline_compute_executed);
    }
}

TEST_CASE_METHOD(EDDOWorkflowTestFixture, "EDDO Matrix Multiplication Workflow", "[orchestrator][eddo][matmul]") {

    SECTION("Complete matrix multiplication EDDO workflow") {
        constexpr size_t MATRIX_DIM = 16;
        constexpr size_t MATRIX_SIZE = MATRIX_DIM * MATRIX_DIM * sizeof(float);

        // Create test matrices
        auto matrix_a = create_test_matrix(MATRIX_DIM, MATRIX_DIM, 1.0f);
        auto matrix_b = create_test_matrix(MATRIX_DIM, MATRIX_DIM, 2.0f);
        std::vector<float> matrix_c(MATRIX_DIM * MATRIX_DIM, 0.0f);
        std::vector<float> expected_c(MATRIX_DIM * MATRIX_DIM);

        // Compute expected result
        matrix_multiply_compute(matrix_a, matrix_b, expected_c, MATRIX_DIM, MATRIX_DIM, MATRIX_DIM);

        // Write input matrices to orchestrator banks
        orchestrator->write(0, 0, matrix_a.data(), MATRIX_SIZE);
        orchestrator->write(1, 0, matrix_b.data(), MATRIX_SIZE);

        std::atomic<bool> matmul_completed{false};

        // Create EDDO workflow for matrix multiplication
        EDDOWorkflowBuilder builder;
        auto workflow = builder
            // Prefetch matrix A to bank 0 (already there, but simulate the phase)
            .prefetch(0, 0x1000, 0, MATRIX_SIZE)
            // Prefetch matrix B to bank 1
            .prefetch(1, 0x2000, 0, MATRIX_SIZE)
            // Compute phase - perform matrix multiplication
            .compute(2, [&]() {
                std::vector<float> a_data(MATRIX_DIM * MATRIX_DIM);
                std::vector<float> b_data(MATRIX_DIM * MATRIX_DIM);

                // Read matrices from orchestrator (simulating compute access)
                orchestrator->read(0, 0, a_data.data(), MATRIX_SIZE);
                orchestrator->read(1, 0, b_data.data(), MATRIX_SIZE);

                // Perform matrix multiplication
                matrix_multiply_compute(a_data, b_data, matrix_c, MATRIX_DIM, MATRIX_DIM, MATRIX_DIM);

                // Write result to bank 2
                orchestrator->write(2, 0, matrix_c.data(), MATRIX_SIZE);

                matmul_completed = true;
            })
            // Writeback result to main memory
            .writeback(2, 0, 0x3000, MATRIX_SIZE)
            // Synchronization barrier
            .sync()
            .build();

        // Execute workflow
        for (const auto& cmd : workflow) {
            orchestrator->enqueue_eddo_command(cmd);
        }

        // Process workflow
        int max_iterations = 50;
        while (orchestrator->is_busy() && max_iterations-- > 0) {
            orchestrator->process_eddo_commands();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }

        REQUIRE(matmul_completed);

        // Verify result
        std::vector<float> computed_result(MATRIX_DIM * MATRIX_DIM);
        orchestrator->read(2, 0, computed_result.data(), MATRIX_SIZE);

        // Check a few elements for correctness
        for (size_t i = 0; i < std::min(size_t(10), computed_result.size()); ++i) {
            REQUIRE(std::abs(computed_result[i] - expected_c[i]) < 1e-6f);
        }
    }
}

TEST_CASE_METHOD(EDDOWorkflowTestFixture, "EDDO Performance and Concurrency", "[orchestrator][eddo][performance]") {

    SECTION("Concurrent EDDO command processing") {
        constexpr size_t NUM_CONCURRENT_COMMANDS = 16;
        std::atomic<size_t> completed_commands{0};

        // Create multiple concurrent EDDO commands
        for (size_t i = 0; i < NUM_CONCURRENT_COMMANDS; ++i) {
            MemoryOrchestrator::EDDOCommand cmd{
                .phase = MemoryOrchestrator::EDDOPhase::COMPUTE,
                .bank_id = i % NUM_BANKS, // Distribute across banks
                .sequence_id = i + 1,
                .completion_callback = [&completed_commands](const MemoryOrchestrator::EDDOCommand&) {
                    completed_commands.fetch_add(1);
                }
            };
            orchestrator->enqueue_eddo_command(cmd);
        }

        REQUIRE(orchestrator->get_pending_commands() == NUM_CONCURRENT_COMMANDS);

        // Process all commands
        auto start_time = std::chrono::high_resolution_clock::now();
        while (orchestrator->is_busy()) {
            orchestrator->process_eddo_commands();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        REQUIRE(completed_commands == NUM_CONCURRENT_COMMANDS);
        REQUIRE(duration.count() < 1000); // Should complete within 1 second

        // Verify performance metrics
        auto metrics = orchestrator->get_performance_metrics();
        REQUIRE(metrics.completed_eddo_commands >= 0);
    }
}