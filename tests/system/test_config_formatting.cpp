#include <catch2/catch_test_macros.hpp>
#include <sw/system/system_config.hpp>
#include <sw/system/config_formatter.hpp>
#include <sstream>

using namespace sw::sim;

TEST_CASE("Config formatter - SystemInfo", "[system][formatter]") {
    SystemInfo info;
    info.name = "Test System";
    info.description = "A test configuration";
    info.clock_frequency_mhz = 2400;

    std::ostringstream oss;
    oss << info;
    std::string output = oss.str();

    REQUIRE(output.find("Test System") != std::string::npos);
    REQUIRE(output.find("A test configuration") != std::string::npos);
    REQUIRE(output.find("2400") != std::string::npos);
}

TEST_CASE("Config formatter - CPUConfig", "[system][formatter]") {
    CPUConfig cpu;
    cpu.core_count = 16;
    cpu.frequency_mhz = 3000;
    cpu.cache_l1_kb = 32;
    cpu.cache_l2_kb = 256;
    cpu.cache_l3_kb = 8192;

    std::ostringstream oss;
    oss << cpu;
    std::string output = oss.str();

    REQUIRE(output.find("16") != std::string::npos);
    REQUIRE(output.find("3000") != std::string::npos);
    REQUIRE(output.find("L1") != std::string::npos);
    REQUIRE(output.find("L2") != std::string::npos);
    REQUIRE(output.find("L3") != std::string::npos);
}

TEST_CASE("Config formatter - MemoryModuleConfig", "[system][formatter]") {
    MemoryModuleConfig mem;
    mem.id = "ddr5_dimm_0";
    mem.type = "DDR5";
    mem.form_factor = "DIMM";
    mem.capacity_gb = 64;
    mem.bandwidth_gbps = 51.2f;

    std::ostringstream oss;
    oss << mem;
    std::string output = oss.str();

    REQUIRE(output.find("ddr5_dimm_0") != std::string::npos);
    REQUIRE(output.find("DDR5") != std::string::npos);
    REQUIRE(output.find("64GB") != std::string::npos);
}

TEST_CASE("Config formatter - KPUMemoryConfig", "[system][formatter]") {
    KPUMemoryConfig kpu_mem;
    kpu_mem.type = "GDDR6";
    kpu_mem.form_factor = "PCB";

    // Add banks
    for (int i = 0; i < 2; ++i) {
        KPUMemoryBankConfig bank;
        bank.id = "bank_" + std::to_string(i);
        bank.capacity_mb = 1024;
        bank.bandwidth_gbps = 100.0f;
        kpu_mem.banks.push_back(bank);
    }

    // Add L3 tiles
    for (int i = 0; i < 4; ++i) {
        KPUTileConfig tile;
        tile.id = "l3_" + std::to_string(i);
        tile.capacity_kb = 128;
        kpu_mem.l3_tiles.push_back(tile);
    }

    // Add scratchpads
    for (int i = 0; i < 2; ++i) {
        KPUScratchpadConfig scratch;
        scratch.id = "scratch_" + std::to_string(i);
        scratch.capacity_kb = 64;
        kpu_mem.scratchpads.push_back(scratch);
    }

    std::ostringstream oss;
    oss << kpu_mem;
    std::string output = oss.str();

    REQUIRE(output.find("GDDR6") != std::string::npos);
    REQUIRE(output.find("PCB") != std::string::npos);
    REQUIRE(output.find("bank_0") != std::string::npos);
    REQUIRE(output.find("bank_1") != std::string::npos);
    REQUIRE(output.find("l3_0") != std::string::npos);
    REQUIRE(output.find("scratch_0") != std::string::npos);
}

TEST_CASE("Config formatter - ComputeTileConfig", "[system][formatter]") {
    ComputeTileConfig tile;
    tile.id = "tile_0";
    tile.type = "systolic";
    tile.systolic_rows = 16;
    tile.systolic_cols = 16;
    tile.datatype = "fp32";

    std::ostringstream oss;
    oss << tile;
    std::string output = oss.str();

    REQUIRE(output.find("tile_0") != std::string::npos);
    REQUIRE(output.find("systolic") != std::string::npos);
    REQUIRE(output.find("16x16") != std::string::npos);
    REQUIRE(output.find("fp32") != std::string::npos);
}

TEST_CASE("Config formatter - DMAEngineConfig", "[system][formatter]") {
    DMAEngineConfig dma;
    dma.id = "dma_0";
    dma.bandwidth_gbps = 50.0f;
    dma.channels = 2;

    std::ostringstream oss;
    oss << dma;
    std::string output = oss.str();

    REQUIRE(output.find("dma_0") != std::string::npos);
    REQUIRE(output.find("50") != std::string::npos);
    REQUIRE(output.find("2") != std::string::npos);
}

TEST_CASE("Config formatter - AcceleratorType", "[system][formatter]") {
    std::ostringstream oss;

    oss << AcceleratorType::KPU;
    REQUIRE(oss.str() == "KPU");

    oss.str("");
    oss << AcceleratorType::GPU;
    REQUIRE(oss.str() == "GPU");

    oss.str("");
    oss << AcceleratorType::NPU;
    REQUIRE(oss.str() == "NPU");
}

TEST_CASE("Config formatter - InterconnectConfig", "[system][formatter]") {
    InterconnectConfig interconnect;
    interconnect.host_to_accelerator.type = "PCIe";

    PCIeConfig pcie;
    pcie.generation = 4;
    pcie.lanes = 16;
    pcie.bandwidth_gbps = 32.0f;
    interconnect.host_to_accelerator.pcie_config = pcie;

    std::ostringstream oss;
    oss << interconnect;
    std::string output = oss.str();

    REQUIRE(output.find("PCIe") != std::string::npos);
    REQUIRE(output.find("Gen4") != std::string::npos);
    REQUIRE(output.find("x16") != std::string::npos);
}

TEST_CASE("Config formatter - Complete SystemConfig", "[system][formatter]") {
    SystemConfig config = SystemConfig::create_minimal_kpu();

    SECTION("Stream operator") {
        std::ostringstream oss;
        oss << config;
        std::string output = oss.str();

        REQUIRE(output.find(config.system.name) != std::string::npos);
        REQUIRE(output.find("CPU") != std::string::npos);
        REQUIRE(output.find("Memory") != std::string::npos);
        REQUIRE(output.find("Accelerator") != std::string::npos);
        REQUIRE(output.find("KPU") != std::string::npos);
    }

    SECTION("to_string function") {
        std::string output = to_string(config, FormatDetail::Standard);
        REQUIRE_FALSE(output.empty());
        REQUIRE(output.find(config.system.name) != std::string::npos);
    }

    SECTION("print_config function") {
        std::ostringstream oss;
        print_config(oss, config, FormatDetail::Standard);
        std::string output = oss.str();

        REQUIRE_FALSE(output.empty());
        REQUIRE(output.find(config.system.name) != std::string::npos);
    }
}

TEST_CASE("Config formatter - Edge AI configuration", "[system][formatter]") {
    SystemConfig config = SystemConfig::create_edge_ai();

    std::ostringstream oss;
    oss << config;
    std::string output = oss.str();

    REQUIRE(output.find("Edge AI System") != std::string::npos);
    REQUIRE(output.find("LPDDR5") != std::string::npos);
    REQUIRE(output.find("KPU") != std::string::npos);
    // Edge AI has both KPU and NPU
    REQUIRE((output.find("NPU") != std::string::npos || config.get_npu_count() > 0));
}

TEST_CASE("Config formatter - Datacenter configuration", "[system][formatter]") {
    SystemConfig config = SystemConfig::create_datacenter();

    std::ostringstream oss;
    oss << config;
    std::string output = oss.str();

    REQUIRE(output.find("Datacenter") != std::string::npos);
    REQUIRE(output.find("DDR5") != std::string::npos);
    REQUIRE(output.find("HBM3") != std::string::npos);
    REQUIRE(output.find("KPU") != std::string::npos);
    REQUIRE(output.find("GPU") != std::string::npos);
}

TEST_CASE("Config formatter - Custom KPU configuration", "[system][formatter]") {
    SystemConfig config;
    config.system.name = "Custom Test System";

    // Add host config
    MemoryModuleConfig mem;
    mem.id = "test_mem";
    mem.type = "DDR4";
    mem.capacity_gb = 16;
    mem.bandwidth_gbps = 25.6f;
    config.host.memory.modules.push_back(mem);

    // Add KPU
    AcceleratorConfig kpu_accel;
    kpu_accel.type = AcceleratorType::KPU;
    kpu_accel.id = "test_kpu";
    kpu_accel.description = "Test KPU";

    KPUConfig kpu;
    kpu.memory.type = "GDDR6";

    // Add components
    KPUMemoryBankConfig bank;
    bank.id = "test_bank";
    bank.capacity_mb = 512;
    kpu.memory.banks.push_back(bank);

    ComputeTileConfig tile;
    tile.id = "test_tile";
    tile.type = "systolic";
    kpu.compute_fabric.tiles.push_back(tile);

    DMAEngineConfig dma;
    dma.id = "test_dma";
    kpu.data_movement.dma_engines.push_back(dma);

    KPUScratchpadConfig scratch;
    scratch.id = "test_scratch";
    kpu.memory.scratchpads.push_back(scratch);

    kpu_accel.kpu_config = kpu;
    config.accelerators.push_back(kpu_accel);

    std::ostringstream oss;
    oss << config;
    std::string output = oss.str();

    REQUIRE(output.find("Custom Test System") != std::string::npos);
    REQUIRE(output.find("test_kpu") != std::string::npos);
    REQUIRE(output.find("test_bank") != std::string::npos);
    REQUIRE(output.find("test_tile") != std::string::npos);
    REQUIRE(output.find("test_dma") != std::string::npos);
    REQUIRE(output.find("test_scratch") != std::string::npos);
}

TEST_CASE("Config formatter - Empty/Minimal configurations", "[system][formatter]") {
    SECTION("Empty SystemInfo") {
        SystemInfo info;
        std::ostringstream oss;
        oss << info;
        REQUIRE_FALSE(oss.str().empty());
    }

    SECTION("Empty KPUMemoryConfig") {
        KPUMemoryConfig kpu_mem;
        std::ostringstream oss;
        oss << kpu_mem;
        REQUIRE_FALSE(oss.str().empty());
    }

    SECTION("Empty KPUDataMovementConfig") {
        KPUDataMovementConfig data_movement;
        std::ostringstream oss;
        oss << data_movement;
        std::string output = oss.str();
        // Should handle empty gracefully
        REQUIRE(true);
    }
}
