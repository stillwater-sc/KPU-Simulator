#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <sw/kpu/components/buffet.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <thread>

using namespace sw::kpu;

// Performance test fixture
class BuffetPerformanceFixture {
public:
    static constexpr size_t NUM_BANKS = 8;
    static constexpr size_t BANK_SIZE_KB = 256;
    static constexpr size_t LARGE_DATA_SIZE = 64 * 1024; // 64KB

    Buffet::BankConfig performance_config{
        .bank_size_kb = BANK_SIZE_KB,
        .cache_line_size = 64,
        .num_ports = 4,
        .access_pattern = Buffet::AccessPattern::SEQUENTIAL,
        .enable_prefetch = true
    };

    std::unique_ptr<Buffet> buffet;
    std::vector<uint8_t> large_test_data;
    std::mt19937 rng;

    BuffetPerformanceFixture() : rng(std::random_device{}()) {
        buffet = std::make_unique<Buffet>(0, NUM_BANKS, performance_config);

        // Generate large test dataset
        large_test_data.resize(LARGE_DATA_SIZE);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& byte : large_test_data) {
            byte = static_cast<uint8_t>(dist(rng));
        }
    }

    // Generate random access pattern
    std::vector<std::pair<size_t, Address>> generate_random_accesses(size_t count) {
        std::vector<std::pair<size_t, Address>> accesses;
        std::uniform_int_distribution<size_t> bank_dist(0, NUM_BANKS - 1);
        std::uniform_int_distribution<Address> addr_dist(0, BANK_SIZE_KB * 1024 - LARGE_DATA_SIZE);

        for (size_t i = 0; i < count; ++i) {
            accesses.emplace_back(bank_dist(rng), addr_dist(rng));
        }
        return accesses;
    }
};

TEST_CASE_METHOD(BuffetPerformanceFixture, "Buffet Performance Benchmarks", "[buffet][performance][!benchmark]") {

    SECTION("Sequential read/write performance") {
        constexpr size_t NUM_OPERATIONS = 100;
        constexpr size_t OPERATION_SIZE = 4096; // 4KB chunks

        BENCHMARK("Sequential write operations") {
            for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
                size_t bank_id = i % NUM_BANKS;
                Address addr = (i * OPERATION_SIZE) % (BANK_SIZE_KB * 1024 - OPERATION_SIZE);
                buffet->write(bank_id, addr, large_test_data.data(), OPERATION_SIZE);
            }
        };

        BENCHMARK("Sequential read operations") {
            std::vector<uint8_t> read_buffer(OPERATION_SIZE);
            for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
                size_t bank_id = i % NUM_BANKS;
                Address addr = (i * OPERATION_SIZE) % (BANK_SIZE_KB * 1024 - OPERATION_SIZE);
                buffet->read(bank_id, addr, read_buffer.data(), OPERATION_SIZE);
            }
        };
    }

    SECTION("Random access performance") {
        constexpr size_t NUM_RANDOM_OPERATIONS = 50;
        constexpr size_t RANDOM_OPERATION_SIZE = 1024;

        auto random_accesses = generate_random_accesses(NUM_RANDOM_OPERATIONS);

        BENCHMARK("Random write operations") {
            for (const auto& [bank_id, addr] : random_accesses) {
                buffet->write(bank_id, addr, large_test_data.data(), RANDOM_OPERATION_SIZE);
            }
        };

        BENCHMARK("Random read operations") {
            std::vector<uint8_t> read_buffer(RANDOM_OPERATION_SIZE);
            for (const auto& [bank_id, addr] : random_accesses) {
                buffet->read(bank_id, addr, read_buffer.data(), RANDOM_OPERATION_SIZE);
            }
        };
    }
}

TEST_CASE_METHOD(BuffetPerformanceFixture, "EDDO Performance Benchmarks", "[buffet][eddo][performance][!benchmark]") {

    SECTION("EDDO command processing throughput") {
        constexpr size_t NUM_COMMANDS = 1000;

        // Pre-generate EDDO commands
        std::vector<Buffet::EDDOCommand> commands;
        commands.reserve(NUM_COMMANDS);

        for (size_t i = 0; i < NUM_COMMANDS; ++i) {
            commands.push_back(Buffet::EDDOCommand{
                .phase = Buffet::EDDOPhase::COMPUTE,
                .bank_id = i % NUM_BANKS,
                .sequence_id = i + 1
            });
        }

        BENCHMARK("EDDO command enqueuing") {
            buffet->reset(); // Start fresh
            for (const auto& cmd : commands) {
                buffet->enqueue_eddo_command(cmd);
            }
        };

        BENCHMARK("EDDO command processing") {
            buffet->reset();
            for (const auto& cmd : commands) {
                buffet->enqueue_eddo_command(cmd);
            }

            while (buffet->is_busy()) {
                buffet->process_eddo_commands();
            }
        };
    }

    SECTION("Complex EDDO workflow performance") {
        constexpr size_t MATRIX_SIZE = 64; // 64x64 matrix
        constexpr size_t MATRIX_BYTES = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

        BENCHMARK("Matrix multiplication EDDO workflow") {
            buffet->reset();

            std::atomic<bool> compute_done{false};

            EDDOWorkflowBuilder builder;
            builder.prefetch(0, 0x10000, 0, MATRIX_BYTES)
                   .prefetch(1, 0x20000, 0, MATRIX_BYTES)
                   .compute(2, [&compute_done]() {
                       // Simulate matrix computation work
                       std::this_thread::sleep_for(std::chrono::microseconds(100));
                       compute_done = true;
                   })
                   .writeback(2, 0, 0x30000, MATRIX_BYTES)
                   .sync()
                   .execute_on(*buffet);

            while (buffet->is_busy()) {
                buffet->process_eddo_commands();
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        };
    }
}

TEST_CASE_METHOD(BuffetPerformanceFixture, "Buffet Scalability Tests", "[buffet][scalability][performance]") {

    SECTION("Bank contention under load") {
        constexpr size_t NUM_THREADS = 4;
        constexpr size_t OPERATIONS_PER_THREAD = 100;
        constexpr size_t OPERATION_SIZE = 1024;

        auto worker = [&](size_t thread_id) {
            std::vector<uint8_t> thread_data(OPERATION_SIZE, static_cast<uint8_t>(thread_id));

            for (size_t i = 0; i < OPERATIONS_PER_THREAD; ++i) {
                size_t bank_id = (thread_id * OPERATIONS_PER_THREAD + i) % NUM_BANKS;
                Address addr = i * OPERATION_SIZE;

                try {
                    buffet->write(bank_id, addr, thread_data.data(), OPERATION_SIZE);

                    std::vector<uint8_t> read_back(OPERATION_SIZE);
                    buffet->read(bank_id, addr, read_back.data(), OPERATION_SIZE);
                } catch (const std::exception&) {
                    // Handle bank contention gracefully
                }
            }
        };

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        for (size_t i = 0; i < NUM_THREADS; ++i) {
            threads.emplace_back(worker, i);
        }

        for (auto& t : threads) {
            t.join();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        REQUIRE(duration.count() < 5000); // Should complete within 5 seconds

        // Check performance metrics
        auto metrics = buffet->get_performance_metrics();
        REQUIRE(metrics.total_read_accesses > 0);
        REQUIRE(metrics.total_write_accesses > 0);
    }

    SECTION("Memory usage scaling") {
        // Test different bank configurations
        std::vector<size_t> bank_counts = {2, 4, 8, 16};
        std::vector<size_t> bank_sizes_kb = {32, 64, 128, 256};

        for (size_t num_banks : bank_counts) {
            for (size_t bank_size : bank_sizes_kb) {
                Buffet::BankConfig config{
                    .bank_size_kb = bank_size,
                    .cache_line_size = 64,
                    .num_ports = 2,
                    .access_pattern = Buffet::AccessPattern::SEQUENTIAL,
                    .enable_prefetch = true
                };

                auto test_buffet = std::make_unique<Buffet>(1, num_banks, config);

                // Verify configuration
                REQUIRE(test_buffet->get_num_banks() == num_banks);

                for (size_t i = 0; i < num_banks; ++i) {
                    REQUIRE(test_buffet->get_bank_capacity(i) == bank_size * 1024);
                }

                // Test basic operations
                constexpr size_t TEST_SIZE = 1024;
                std::vector<uint8_t> test_data(TEST_SIZE, 0xCC);

                if (bank_size * 1024 >= TEST_SIZE) {
                    REQUIRE_NOTHROW(test_buffet->write(0, 0, test_data.data(), TEST_SIZE));

                    std::vector<uint8_t> read_data(TEST_SIZE);
                    REQUIRE_NOTHROW(test_buffet->read(0, 0, read_data.data(), TEST_SIZE));

                    REQUIRE(test_data == read_data);
                }
            }
        }
    }
}

TEST_CASE_METHOD(BuffetPerformanceFixture, "EDDO vs Direct Access Performance", "[buffet][comparison][performance]") {

    SECTION("Compare EDDO orchestrated vs direct memory access") {
        constexpr size_t OPERATION_COUNT = 100;
        constexpr size_t DATA_SIZE = 4096;

        std::vector<uint8_t> test_data(DATA_SIZE);
        std::iota(test_data.begin(), test_data.end(), 0);

        // Benchmark direct access
        auto direct_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < OPERATION_COUNT; ++i) {
            size_t bank_id = i % NUM_BANKS;
            Address addr = (i * 128) % (BANK_SIZE_KB * 1024 - DATA_SIZE);

            buffet->write(bank_id, addr, test_data.data(), DATA_SIZE);

            std::vector<uint8_t> read_buffer(DATA_SIZE);
            buffet->read(bank_id, addr, read_buffer.data(), DATA_SIZE);
        }
        auto direct_end = std::chrono::high_resolution_clock::now();
        auto direct_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            direct_end - direct_start);

        // Benchmark EDDO orchestrated access
        buffet->reset();

        std::atomic<size_t> eddo_operations_completed{0};

        auto eddo_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < OPERATION_COUNT; ++i) {
            size_t bank_id = i % NUM_BANKS;

            Buffet::EDDOCommand cmd{
                .phase = Buffet::EDDOPhase::COMPUTE,
                .bank_id = bank_id,
                .sequence_id = i + 1,
                .completion_callback = [&eddo_operations_completed](const Buffet::EDDOCommand&) {
                    eddo_operations_completed.fetch_add(1);
                }
            };
            buffet->enqueue_eddo_command(cmd);
        }

        while (buffet->is_busy()) {
            buffet->process_eddo_commands();
        }
        auto eddo_end = std::chrono::high_resolution_clock::now();
        auto eddo_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            eddo_end - eddo_start);

        REQUIRE(eddo_operations_completed == OPERATION_COUNT);

        // EDDO might have overhead but should be reasonable
        double overhead_ratio = static_cast<double>(eddo_duration.count()) / direct_duration.count();
        REQUIRE(overhead_ratio < 10.0); // Should not be more than 10x slower

        INFO("Direct access time: " << direct_duration.count() << " microseconds");
        INFO("EDDO access time: " << eddo_duration.count() << " microseconds");
        INFO("Overhead ratio: " << overhead_ratio);
    }
}