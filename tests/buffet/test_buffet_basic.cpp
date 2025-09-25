#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/kpu/components/buffet.hpp>
#include <vector>
#include <algorithm>
#include <cstring>
#include <thread>
#include <chrono>

using namespace sw::kpu;

// Test fixture for Buffet tests
class BuffetTestFixture {
public:
    static constexpr size_t NUM_BANKS = 4;
    static constexpr size_t BANK_SIZE_KB = 64;

    Buffet::BankConfig default_config{
        .bank_size_kb = BANK_SIZE_KB,
        .cache_line_size = 64,
        .num_ports = 2,
        .access_pattern = Buffet::AccessPattern::SEQUENTIAL,
        .enable_prefetch = true
    };

    std::unique_ptr<Buffet> buffet;

    BuffetTestFixture() {
        buffet = std::make_unique<Buffet>(0, NUM_BANKS, default_config);
    }

    // Helper to generate test data
    std::vector<uint8_t> generate_test_data(size_t size, uint8_t pattern = 0xAA) {
        std::vector<uint8_t> data(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<uint8_t>((pattern + i) & 0xFF);
        }
        return data;
    }

    // Helper to verify data integrity
    bool verify_data(const std::vector<uint8_t>& expected,
                     const std::vector<uint8_t>& actual) {
        return expected.size() == actual.size() &&
               std::equal(expected.begin(), expected.end(), actual.begin());
    }
};

TEST_CASE_METHOD(BuffetTestFixture, "Buffet Basic Functionality", "[buffet][basic]") {

    SECTION("Buffet initialization") {
        REQUIRE(buffet->get_num_banks() == NUM_BANKS);
        REQUIRE(buffet->get_buffet_id() == 0);

        for (size_t i = 0; i < NUM_BANKS; ++i) {
            REQUIRE(buffet->get_bank_capacity(i) == BANK_SIZE_KB * 1024);
            REQUIRE(buffet->get_bank_occupancy(i) == 0);
            REQUIRE(buffet->is_ready(i));
            REQUIRE_FALSE(buffet->is_bank_busy(i));
            REQUIRE(buffet->get_bank_phase(i) == Buffet::EDDOPhase::SYNC);
        }
    }

    SECTION("Basic read/write operations") {
        constexpr size_t TEST_SIZE = 1024;
        auto test_data = generate_test_data(TEST_SIZE, 0x42);
        std::vector<uint8_t> read_buffer(TEST_SIZE);

        // Write data to bank 0
        REQUIRE_NOTHROW(buffet->write(0, 0, test_data.data(), TEST_SIZE));
        REQUIRE(buffet->get_bank_occupancy(0) == TEST_SIZE);

        // Read data back from bank 0
        REQUIRE_NOTHROW(buffet->read(0, 0, read_buffer.data(), TEST_SIZE));
        REQUIRE(verify_data(test_data, read_buffer));
    }

    SECTION("Multi-bank operations") {
        constexpr size_t TEST_SIZE = 512;

        for (size_t bank_id = 0; bank_id < NUM_BANKS; ++bank_id) {
            auto test_data = generate_test_data(TEST_SIZE, static_cast<uint8_t>(bank_id));

            REQUIRE_NOTHROW(buffet->write(bank_id, 0, test_data.data(), TEST_SIZE));
            REQUIRE(buffet->get_bank_occupancy(bank_id) == TEST_SIZE);
        }

        // Verify each bank contains correct data
        for (size_t bank_id = 0; bank_id < NUM_BANKS; ++bank_id) {
            auto expected_data = generate_test_data(TEST_SIZE, static_cast<uint8_t>(bank_id));
            std::vector<uint8_t> read_buffer(TEST_SIZE);

            REQUIRE_NOTHROW(buffet->read(bank_id, 0, read_buffer.data(), TEST_SIZE));
            REQUIRE(verify_data(expected_data, read_buffer));
        }
    }

    SECTION("Bank configuration") {
        Buffet::BankConfig custom_config{
            .bank_size_kb = 128,
            .cache_line_size = 128,
            .num_ports = 4,
            .access_pattern = Buffet::AccessPattern::RANDOM,
            .enable_prefetch = false
        };

        REQUIRE_NOTHROW(buffet->configure_bank(0, custom_config));
        REQUIRE(buffet->get_bank_capacity(0) == 128 * 1024);
    }

    SECTION("Error handling") {
        constexpr size_t INVALID_BANK = 999;
        constexpr size_t TEST_SIZE = 100;
        auto test_data = generate_test_data(TEST_SIZE);

        // Invalid bank ID
        REQUIRE_THROWS(buffet->write(INVALID_BANK, 0, test_data.data(), TEST_SIZE));
        REQUIRE_THROWS(buffet->read(INVALID_BANK, 0, test_data.data(), TEST_SIZE));
        REQUIRE_FALSE(buffet->is_ready(INVALID_BANK));

        // Address out of bounds
        Size bank_capacity = buffet->get_bank_capacity(0);
        REQUIRE_THROWS(buffet->write(0, bank_capacity, test_data.data(), TEST_SIZE));
        REQUIRE_THROWS(buffet->read(0, bank_capacity, test_data.data(), TEST_SIZE));
    }
}

TEST_CASE_METHOD(BuffetTestFixture, "Buffet Reset and Cleanup", "[buffet][cleanup]") {

    SECTION("Reset functionality") {
        constexpr size_t TEST_SIZE = 512;
        auto test_data = generate_test_data(TEST_SIZE);

        // Write data to all banks
        for (size_t bank_id = 0; bank_id < NUM_BANKS; ++bank_id) {
            buffet->write(bank_id, 0, test_data.data(), TEST_SIZE);
        }

        // Verify data is written
        for (size_t bank_id = 0; bank_id < NUM_BANKS; ++bank_id) {
            REQUIRE(buffet->get_bank_occupancy(bank_id) == TEST_SIZE);
        }

        // Reset buffet
        buffet->reset();

        // Verify all banks are reset
        for (size_t bank_id = 0; bank_id < NUM_BANKS; ++bank_id) {
            REQUIRE(buffet->get_bank_occupancy(bank_id) == 0);
            REQUIRE(buffet->is_ready(bank_id));
            REQUIRE(buffet->get_bank_phase(bank_id) == Buffet::EDDOPhase::SYNC);
        }
    }

    SECTION("Flush all banks") {
        constexpr size_t TEST_SIZE = 256;
        auto test_data = generate_test_data(TEST_SIZE);

        // Write data to banks
        buffet->write(0, 0, test_data.data(), TEST_SIZE);
        buffet->write(1, 100, test_data.data(), TEST_SIZE);

        REQUIRE(buffet->get_bank_occupancy(0) == TEST_SIZE);
        REQUIRE(buffet->get_bank_occupancy(1) == 356); // 100 + 256

        // Flush all banks
        buffet->flush_all_banks();

        // Verify banks are flushed
        for (size_t bank_id = 0; bank_id < NUM_BANKS; ++bank_id) {
            REQUIRE(buffet->get_bank_occupancy(bank_id) == 0);
        }
    }
}

TEST_CASE_METHOD(BuffetTestFixture, "Performance Metrics", "[buffet][metrics]") {

    SECTION("Basic performance tracking") {
        constexpr size_t TEST_SIZE = 512;
        constexpr size_t NUM_OPERATIONS = 10;
        auto test_data = generate_test_data(TEST_SIZE);
        std::vector<uint8_t> read_buffer(TEST_SIZE);

        // Perform multiple read/write operations
        for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
            buffet->write(0, i * 64, test_data.data(), TEST_SIZE);
            buffet->read(0, i * 64, read_buffer.data(), TEST_SIZE);
        }

        auto metrics = buffet->get_performance_metrics();

        REQUIRE(metrics.total_write_accesses >= NUM_OPERATIONS);
        REQUIRE(metrics.total_read_accesses >= NUM_OPERATIONS);
        REQUIRE(metrics.average_bank_utilization >= 0.0);
        REQUIRE(metrics.average_bank_utilization <= 1.0);
    }
}