#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/memory/memory_map.hpp>
#include <cstring>
#include <vector>
#include <thread>

using namespace sw::kpu::memory;
using namespace sw::kpu;

TEST_CASE("MemoryMap basic functionality", "[memory][memmap]") {
    SECTION("Create and destroy mapping") {
        MemoryMap::Config config(4096);  // 4KB
        config.populate = true;  // Commit memory for direct access
        MemoryMap map(config);

        REQUIRE(map.is_mapped());
        REQUIRE(map.size() == 4096);
        REQUIRE(map.data() != nullptr);
    }

    SECTION("Get system page size") {
        Size page_size = MemoryMap::get_system_page_size();
        REQUIRE(page_size > 0);
        // Typical page sizes are 4KB, 8KB, 16KB
        REQUIRE(page_size >= 4096);
    }

    SECTION("Write and read simple data") {
        MemoryMap::Config config(1024 * 1024);  // 1MB
        config.populate = true;  // Commit memory for direct access
        MemoryMap map(config);

        // Write some data
        uint32_t test_value = 0xDEADBEEF;
        std::memcpy(map.data(), &test_value, sizeof(test_value));

        // Read it back
        uint32_t read_value = 0;
        std::memcpy(&read_value, map.data(), sizeof(read_value));

        REQUIRE(read_value == test_value);
    }

    SECTION("Write and read at different offsets") {
        MemoryMap::Config config(1024 * 1024);  // 1MB
        config.populate = true;  // Commit memory for direct access
        MemoryMap map(config);

        // Write pattern at different offsets
        for (Size offset = 0; offset < 10; ++offset) {
            uint64_t pattern = 0x0123456789ABCDEF + offset;
            std::memcpy(static_cast<char*>(map.data()) + offset * 1024,
                       &pattern, sizeof(pattern));
        }

        // Read back and verify
        for (Size offset = 0; offset < 10; ++offset) {
            uint64_t expected = 0x0123456789ABCDEF + offset;
            uint64_t actual = 0;
            std::memcpy(&actual,
                       static_cast<char*>(map.data()) + offset * 1024,
                       sizeof(actual));
            REQUIRE(actual == expected);
        }
    }
}

TEST_CASE("MemoryMap sparse allocation", "[memory][memmap][sparse]") {
    // NOTE: MAP_NORESERVE sparse allocation with on-demand page faulting does not
    // work reliably on WSL. The prefault() operation (which tries to touch pages
    // to trigger page faults) causes bus errors on WSL.
    //
    // This test works on:
    // - Native Windows (using VirtualAlloc with MEM_RESERVE + explicit MEM_COMMIT in prefault)
    // - Native Linux (using mmap with on-demand paging)
    //
    // It should NOT run on WSL due to known bus errors with MAP_NORESERVE.

#ifdef __linux__
    // Check if we're running under WSL by looking for "microsoft" in uname
    std::string uname_output;
    {
        FILE* pipe = popen("uname -r", "r");
        if (pipe) {
            char buffer[128];
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                uname_output += buffer;
            }
            pclose(pipe);
        }
    }

    // Skip if running under WSL
    if (uname_output.find("microsoft") != std::string::npos ||
        uname_output.find("Microsoft") != std::string::npos ||
        uname_output.find("WSL") != std::string::npos) {
        WARN("Sparse MemoryMap tests skipped on WSL - use SparseMemory wrapper instead");
        return;
    }
#endif

    SECTION("Large virtual size with minimal physical usage") {
        // Request 1GB virtual space - this demonstrates sparse allocation well:
        // - Virtual reservation: 1GB (trivial on 64-bit systems)
        // - Physical commitment: Only the pages we actually touch (~16-64KB depending on page size)
        // - This is the key benefit: massive virtual address space with minimal RAM usage
        Size virtual_size = 1ULL * 1024ULL * 1024ULL * 1024ULL;  // 1GB
        MemoryMap::Config config(virtual_size);
        config.populate = false;  // Sparse allocation

        MemoryMap map(config);
        REQUIRE(map.is_mapped());
        REQUIRE(map.size() == virtual_size);

        // Write to a few pages scattered across the address space
        // We're touching maybe 4-16 pages out of 262,144 pages (for 4KB pages)
        // Physical memory used: ~16-64KB out of 1GB reserved
        Size page_size = map.page_size();
        std::vector<Size> test_offsets = {
            0,
            page_size * 10,
            page_size * 100,
            page_size * 1000,
            page_size * 10000,
            page_size * 100000
        };

        // IMPORTANT: For sparse (non-populated) memory, we MUST prefault pages
        // before writing to them
        for (Size offset : test_offsets) {
            if (offset + sizeof(uint64_t) > virtual_size) continue;
            // Prefault the page containing this offset
            map.prefault(offset, page_size);
        }

        for (Size offset : test_offsets) {
            if (offset + sizeof(uint64_t) > virtual_size) continue;
            uint64_t marker = 0xFEEDFACE00000000ULL + offset;
            std::memcpy(static_cast<char*>(map.data()) + offset,
                       &marker, sizeof(marker));
        }

        // Verify the writes
        for (Size offset : test_offsets) {
            if (offset + sizeof(uint64_t) > virtual_size) continue;
            uint64_t expected = 0xFEEDFACE00000000ULL + offset;
            uint64_t actual = 0;
            std::memcpy(&actual,
                       static_cast<char*>(map.data()) + offset,
                       sizeof(actual));
            REQUIRE(actual == expected);
        }
    }
}

TEST_CASE("MemoryMap move semantics", "[memory][memmap]") {
    SECTION("Move constructor") {
        MemoryMap::Config config(4096);
        config.populate = true;  // Commit memory for direct access
        MemoryMap map1(config);

        // Write data to first map
        uint32_t test_val = 0xCAFEBABE;
        std::memcpy(map1.data(), &test_val, sizeof(test_val));

        // Move to second map
        MemoryMap map2(std::move(map1));

        REQUIRE(map2.is_mapped());
        REQUIRE(!map1.is_mapped());

        // Verify data is preserved
        uint32_t read_val = 0;
        std::memcpy(&read_val, map2.data(), sizeof(read_val));
        REQUIRE(read_val == test_val);
    }

    SECTION("Move assignment") {
        MemoryMap::Config config1(4096);
        config1.populate = true;  // Commit memory for direct access
        MemoryMap map1(config1);

        uint32_t test_val1 = 0xDEADBEEF;
        std::memcpy(map1.data(), &test_val1, sizeof(test_val1));

        MemoryMap::Config config2(8192);
        config2.populate = true;  // Commit memory for direct access
        MemoryMap map2(config2);

        // Move assign
        map2 = std::move(map1);

        REQUIRE(map2.is_mapped());
        REQUIRE(!map1.is_mapped());
        REQUIRE(map2.size() == 4096);

        // Verify data
        uint32_t read_val = 0;
        std::memcpy(&read_val, map2.data(), sizeof(read_val));
        REQUIRE(read_val == test_val1);
    }
}

TEST_CASE("MemoryMap statistics", "[memory][memmap][stats]") {
    SECTION("Get statistics") {
        Size size = 1024 * 1024;  // 1MB
        MemoryMap::Config config(size);
        config.populate = true;  // Commit memory
        MemoryMap map(config);

        MemoryMap::Stats stats = map.get_stats();

        REQUIRE(stats.virtual_size == size);
        REQUIRE(stats.page_size > 0);
        REQUIRE(stats.resident_size <= size);
    }
}

TEST_CASE("MemoryMap advice and prefaulting", "[memory][memmap][advice]") {
    SECTION("Prefault pages") {
        Size size = 1024 * 1024;  // 1MB
        MemoryMap::Config config(size);
        config.populate = false;  // Sparse

        MemoryMap map(config);

        // Prefault first 100 pages
        Size page_size = map.page_size();
        REQUIRE_NOTHROW(map.prefault(0, page_size * 100));
    }

    SECTION("Memory advice hints") {
        Size size = 1024 * 1024;  // 1MB
        MemoryMap::Config config(size);
        config.populate = true;  // Commit memory
        MemoryMap map(config);

        // These should not throw
        REQUIRE_NOTHROW(map.advise(0, size, MemoryMap::Advice::Sequential));
        REQUIRE_NOTHROW(map.advise(0, size, MemoryMap::Advice::Random));
        REQUIRE_NOTHROW(map.advise(0, size, MemoryMap::Advice::WillNeed));
    }
}

TEST_CASE("MemoryMap error handling", "[memory][memmap][error]") {
    SECTION("Zero size mapping") {
        MemoryMap::Config config(0);
        REQUIRE_THROWS_AS(MemoryMap(config), std::invalid_argument);
    }

    SECTION("Out of range advice") {
        Size size = 4096;
        MemoryMap::Config config(size);
        config.populate = true;  // Commit memory
        MemoryMap map(config);

        // Advice beyond mapping size should throw
        REQUIRE_THROWS_AS(map.advise(0, size + 1, MemoryMap::Advice::Normal),
                         std::out_of_range);
    }
}

TEST_CASE("MemoryMap concurrent access", "[memory][memmap][concurrent]") {
    SECTION("Multiple threads writing to different pages") {
        Size size = 1024 * 1024;  // 1MB
        MemoryMap::Config config(size);
        config.populate = true;  // Commit memory for direct access
        MemoryMap map(config);

        Size page_size = map.page_size();
        const int num_threads = 4;

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&map, page_size, i]() {
                // Each thread writes to its own set of pages
                Size base_offset = i * 64 * 1024;  // 64KB apart
                for (int j = 0; j < 10; ++j) {
                    uint64_t value = (static_cast<uint64_t>(i) << 32) | j;
                    Size offset = base_offset + j * page_size;
                    if (offset + sizeof(value) <= map.size()) {
                        std::memcpy(static_cast<char*>(map.data()) + offset,
                                   &value, sizeof(value));
                    }
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        // Verify all writes
        for (int i = 0; i < num_threads; ++i) {
            Size base_offset = i * 64 * 1024;
            for (int j = 0; j < 10; ++j) {
                uint64_t expected = (static_cast<uint64_t>(i) << 32) | j;
                uint64_t actual = 0;
                Size offset = base_offset + j * page_size;
                if (offset + sizeof(actual) <= size) {
                    std::memcpy(&actual,
                               static_cast<char*>(map.data()) + offset,
                               sizeof(actual));
                    REQUIRE(actual == expected);
                }
            }
        }
    }
}
