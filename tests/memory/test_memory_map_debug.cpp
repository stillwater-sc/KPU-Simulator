#include <catch2/catch_test_macros.hpp>
#include <sw/memory/memory_map.hpp>
#include <sw/memory/sparse_memory.hpp>
#include <iostream>
#include <cstring>

using namespace sw::kpu::memory;
using namespace sw::kpu;

// Ultra-minimal test to identify segfault location
TEST_CASE("MemoryMap DEBUG - Step by step", "[memory][memmap][debug]") {
    std::cout << "=== Starting DEBUG test ===" << std::endl;

    SECTION("Step 1: Get page size") {
        std::cout << "Step 1: Getting system page size..." << std::endl;
        Size page_size = MemoryMap::get_system_page_size();
        std::cout << "  Page size: " << page_size << " bytes" << std::endl;
        REQUIRE(page_size > 0);
        std::cout << "Step 1: PASSED" << std::endl;
    }

    SECTION("Step 2: Create tiny mapping (4KB, populated)") {
        std::cout << "Step 2: Creating 4KB mapping with populate=true..." << std::endl;

        MemoryMap::Config config(4096);
        config.populate = true;
        std::cout << "  Config created: size=" << config.size
                  << ", populate=" << config.populate << std::endl;

        std::cout << "  Calling MemoryMap constructor..." << std::endl;
        MemoryMap map(config);
        std::cout << "  Constructor returned" << std::endl;

        std::cout << "  Checking is_mapped()..." << std::endl;
        REQUIRE(map.is_mapped());
        std::cout << "  is_mapped() = true" << std::endl;

        std::cout << "  Checking size()..." << std::endl;
        REQUIRE(map.size() == 4096);
        std::cout << "  size() = " << map.size() << std::endl;

        std::cout << "  Checking data()..." << std::endl;
        void* ptr = map.data();
        std::cout << "  data() = " << ptr << std::endl;
        REQUIRE(ptr != nullptr);

        std::cout << "Step 2: PASSED" << std::endl;
    }

    SECTION("Step 3: Write to populated memory") {
        std::cout << "Step 3: Writing to populated memory..." << std::endl;

        MemoryMap::Config config(4096);
        config.populate = true;
        MemoryMap map(config);
        std::cout << "  Map created, data() = " << map.data() << std::endl;

        std::cout << "  Attempting to write uint32_t..." << std::endl;
        uint32_t test_value = 0xDEADBEEF;
        std::cout << "  Test value: 0x" << std::hex << test_value << std::dec << std::endl;

        std::cout << "  Calling memcpy to write..." << std::endl;
        std::memcpy(map.data(), &test_value, sizeof(test_value));
        std::cout << "  Write completed" << std::endl;

        std::cout << "  Attempting to read back..." << std::endl;
        uint32_t read_value = 0;
        std::memcpy(&read_value, map.data(), sizeof(read_value));
        std::cout << "  Read value: 0x" << std::hex << read_value << std::dec << std::endl;

        REQUIRE(read_value == test_value);
        std::cout << "Step 3: PASSED" << std::endl;
    }

    SECTION("Step 4: Create sparse mapping (NO populate)") {
        std::cout << "Step 4: Creating 1MB mapping with populate=false..." << std::endl;

        MemoryMap::Config config(1024 * 1024);  // 1MB
        config.populate = false;  // SPARSE
        std::cout << "  Config: size=" << config.size
                  << ", populate=" << config.populate << std::endl;

        std::cout << "  Calling MemoryMap constructor..." << std::endl;
        MemoryMap map(config);
        std::cout << "  Constructor returned" << std::endl;

        std::cout << "  Map created, data() = " << map.data() << std::endl;
        REQUIRE(map.is_mapped());
        REQUIRE(map.data() != nullptr);

        std::cout << "Step 4: PASSED (created sparse mapping, did NOT write)" << std::endl;
        std::cout << "  NOTE: Writing to sparse memory without commit will crash!" << std::endl;
    }

    SECTION("Step 5: Prefault sparse memory") {
        std::cout << "Step 5: Prefaulting sparse memory..." << std::endl;

        MemoryMap::Config config(1024 * 1024);  // 1MB
        config.populate = false;
        MemoryMap map(config);
        std::cout << "  Sparse map created" << std::endl;

        Size page_size = map.page_size();
        std::cout << "  Prefaulting first 10 pages (" << (page_size * 10) << " bytes)..." << std::endl;

        map.prefault(0, page_size * 10);
        std::cout << "  Prefault completed" << std::endl;

        std::cout << "  Attempting to write to prefaulted region..." << std::endl;
        uint32_t test_value = 0x12345678;
        std::memcpy(map.data(), &test_value, sizeof(test_value));
        std::cout << "  Write succeeded" << std::endl;

        uint32_t read_value = 0;
        std::memcpy(&read_value, map.data(), sizeof(read_value));
        REQUIRE(read_value == test_value);
        std::cout << "  Read succeeded" << std::endl;

        std::cout << "Step 5: PASSED" << std::endl;
    }

    std::cout << "=== DEBUG test completed ===" << std::endl;
}

TEST_CASE("SparseMemory DEBUG - Step by step", "[memory][sparse][debug]") {
    std::cout << "=== Starting SparseMemory DEBUG test ===" << std::endl;

    SECTION("Step 1: Create small sparse memory") {
        std::cout << "Step 1: Creating 1MB sparse memory..." << std::endl;

        SparseMemory::Config config(1024 * 1024);  // 1MB
        std::cout << "  Config: virtual_size=" << config.virtual_size << std::endl;

        std::cout << "  Calling SparseMemory constructor..." << std::endl;
        SparseMemory mem(config);
        std::cout << "  Constructor returned" << std::endl;

        std::cout << "  Checking size()..." << std::endl;
        REQUIRE(mem.size() == 1024 * 1024);
        std::cout << "  size() = " << mem.size() << std::endl;

        std::cout << "  Checking page_size()..." << std::endl;
        Size psize = mem.page_size();
        std::cout << "  page_size() = " << psize << std::endl;
        REQUIRE(psize > 0);

        std::cout << "Step 1: PASSED" << std::endl;
    }

    SECTION("Step 2: Write to sparse memory") {
        std::cout << "Step 2: Writing to sparse memory..." << std::endl;

        SparseMemory::Config config(1024 * 1024);  // 1MB
        SparseMemory mem(config);
        std::cout << "  SparseMemory created" << std::endl;

        std::cout << "  Attempting write at offset 0..." << std::endl;
        uint32_t test_value = 0xABCDEF01;
        mem.write(0, &test_value, sizeof(test_value));
        std::cout << "  Write completed" << std::endl;

        std::cout << "  Attempting read at offset 0..." << std::endl;
        uint32_t read_value = 0;
        mem.read(0, &read_value, sizeof(read_value));
        std::cout << "  Read completed, value: 0x" << std::hex << read_value << std::dec << std::endl;

        REQUIRE(read_value == test_value);
        std::cout << "Step 2: PASSED" << std::endl;
    }

    SECTION("Step 3: Simple write without stats") {
        std::cout << "Step 3: Testing simple writes to sparse memory..." << std::endl;

        // Use 1MB instead of 10MB to avoid overcommit issues on WSL
        SparseMemory::Config config(1024 * 1024);  // 1MB
        std::cout << "  Creating 1MB sparse memory..." << std::endl;
        SparseMemory mem(config);
        std::cout << "  SparseMemory created" << std::endl;

        std::cout << "  Writing to multiple locations..." << std::endl;
        for (int i = 0; i < 3; ++i) {
            Size offset = i * 256 * 1024;  // 256KB apart
            uint32_t value = i + 100;
            std::cout << "    About to write " << value << " at offset " << offset << "..." << std::endl;
            try {
                mem.write(offset, &value, sizeof(value));
                std::cout << "    Wrote " << value << " at offset " << offset << std::endl;

                // Read back to verify
                uint32_t read_value = 0;
                mem.read(offset, &read_value, sizeof(read_value));
                std::cout << "    Read back: " << read_value << std::endl;
                REQUIRE(read_value == value);
            } catch (const std::exception& e) {
                std::cout << "    ERROR: " << e.what() << std::endl;
                throw;
            }
        }

        std::cout << "Step 3: PASSED" << std::endl;
    }

    std::cout << "=== SparseMemory DEBUG test completed ===" << std::endl;
}
