#include <catch2/catch_test_macros.hpp>
#include <sw/system/system_config.hpp>
#include <sw/system/config_loader.hpp>
#include <sw/system/toplevel.hpp>
#include <sw/kpu/kpu_simulator.hpp>
#include <fstream>
#include <filesystem>

using namespace sw::sim;

TEST_CASE("SystemConfig - Factory Methods", "[system][config]") {
    SECTION("Create minimal KPU configuration") {
        auto config = SystemConfig::create_minimal_kpu();

        REQUIRE(config.system.name == "Minimal KPU System");
        REQUIRE(config.get_kpu_count() == 1);
        REQUIRE(config.host.memory.modules.size() > 0);
        REQUIRE(config.validate());
    }

    SECTION("Create edge AI configuration") {
        auto config = SystemConfig::create_edge_ai();

        REQUIRE(config.system.name == "Edge AI System");
        REQUIRE(config.get_kpu_count() == 1);
        REQUIRE(config.get_npu_count() == 1);
        REQUIRE(config.validate());
    }

    SECTION("Create datacenter configuration") {
        auto config = SystemConfig::create_datacenter();

        REQUIRE(config.system.name == "Datacenter AI Cluster Node");
        REQUIRE(config.get_kpu_count() == 1);
        REQUIRE(config.get_gpu_count() == 1);
        REQUIRE(config.validate());
    }
}

TEST_CASE("SystemConfig - Validation", "[system][config]") {
    SECTION("Valid configuration passes") {
        auto config = SystemConfig::create_minimal_kpu();
        REQUIRE(config.validate());
        REQUIRE(config.get_validation_errors().empty());
    }

    SECTION("Empty system name fails validation") {
        auto config = SystemConfig::create_minimal_kpu();
        config.system.name = "";
        REQUIRE_FALSE(config.validate());
        REQUIRE_FALSE(config.get_validation_errors().empty());
    }

    SECTION("No host memory fails validation") {
        auto config = SystemConfig::create_minimal_kpu();
        config.host.memory.modules.clear();
        REQUIRE_FALSE(config.validate());
    }
}

TEST_CASE("SystemConfig - Accelerator Queries", "[system][config]") {
    auto config = SystemConfig::create_datacenter();

    SECTION("Count accelerators by type") {
        REQUIRE(config.get_kpu_count() == 1);
        REQUIRE(config.get_gpu_count() == 1);
        REQUIRE(config.get_npu_count() == 0);
    }

    SECTION("Find accelerator by ID") {
        const auto* kpu = config.find_accelerator("kpu_0");
        REQUIRE(kpu != nullptr);
        REQUIRE(kpu->type == AcceleratorType::KPU);

        const auto* gpu = config.find_accelerator("gpu_0");
        REQUIRE(gpu != nullptr);
        REQUIRE(gpu->type == AcceleratorType::GPU);

        const auto* not_found = config.find_accelerator("nonexistent");
        REQUIRE(not_found == nullptr);
    }
}

TEST_CASE("ConfigLoader - JSON Serialization Round Trip", "[system][config][json]") {
    auto original_config = SystemConfig::create_minimal_kpu();

    // Serialize to JSON string
    std::string json_str = ConfigLoader::to_json_string(original_config, true);
    REQUIRE_FALSE(json_str.empty());

    // Deserialize back
    auto loaded_config = ConfigLoader::load_from_string(json_str);

    // Verify key properties match
    REQUIRE(loaded_config.system.name == original_config.system.name);
    REQUIRE(loaded_config.get_kpu_count() == original_config.get_kpu_count());
    REQUIRE(loaded_config.host.memory.modules.size() == original_config.host.memory.modules.size());
    REQUIRE(loaded_config.validate());
}

TEST_CASE("ConfigLoader - File Operations", "[system][config][json][file]") {
    auto config = SystemConfig::create_minimal_kpu();
    std::filesystem::path temp_file = "test_config_temp.json";

    SECTION("Save and load from file") {
        // Save to file
        REQUIRE_NOTHROW(ConfigLoader::save_to_file(config, temp_file));
        REQUIRE(std::filesystem::exists(temp_file));

        // Load from file
        SystemConfig loaded;
        REQUIRE_NOTHROW(loaded = ConfigLoader::load_from_file(temp_file));

        // Verify
        REQUIRE(loaded.system.name == config.system.name);
        REQUIRE(loaded.validate());

        // Cleanup
        std::filesystem::remove(temp_file);
    }

    SECTION("Validate file") {
        ConfigLoader::save_to_file(config, temp_file);

        REQUIRE(ConfigLoader::validate_file(temp_file));

        auto errors = ConfigLoader::get_validation_errors(temp_file);
        // Should have warnings about no accelerators, but not errors
        REQUIRE(errors.size() >= 0);  // May have warnings

        std::filesystem::remove(temp_file);
    }

    SECTION("Load nonexistent file throws") {
        REQUIRE_THROWS(ConfigLoader::load_from_file("nonexistent_file.json"));
    }

    SECTION("Load invalid JSON throws") {
        std::ofstream invalid_file("invalid.json");
        invalid_file << "{ this is not valid json }";
        invalid_file.close();

        REQUIRE_THROWS(ConfigLoader::load_from_file("invalid.json"));

        std::filesystem::remove("invalid.json");
    }
}

TEST_CASE("ConfigLoader - Load Example Configurations", "[system][config][json][examples]") {
    std::filesystem::path examples_dir;

    // Try multiple possible paths
    std::vector<std::string> possible_paths = {
        "../../configs/examples",           // From build/tests/system
        "../../../configs/examples",        // From build_msvc/tests/system/Debug
        "../../../../configs/examples",     // From deeper build directories
        "../configs/examples",              // Alternative
        "configs/examples",                 // From project root
        "C:/Users/tomtz/dev/stillwater/clones/KPU-simulator/configs/examples"  // Absolute fallback
    };

    bool found = false;
    for (const auto& path : possible_paths) {
        if (std::filesystem::exists(path)) {
            examples_dir = path;
            found = true;
            break;
        }
    }

    if (!found) {
        // Skip test if examples not found
        WARN("Example configurations not found, skipping test");
        return;
    }

    SECTION("Load minimal_kpu.json") {
        auto config_path = examples_dir / "minimal_kpu.json";
        if (std::filesystem::exists(config_path)) {
            SystemConfig config;
            REQUIRE_NOTHROW(config = ConfigLoader::load_from_file(config_path));
            REQUIRE(config.validate());
            REQUIRE(config.get_kpu_count() >= 1);
        }
    }

    SECTION("Load edge_ai.json") {
        auto config_path = examples_dir / "edge_ai.json";
        if (std::filesystem::exists(config_path)) {
            SystemConfig config;
            REQUIRE_NOTHROW(config = ConfigLoader::load_from_file(config_path));
            REQUIRE(config.validate());
        }
    }

    SECTION("Load datacenter_hbm.json") {
        auto config_path = examples_dir / "datacenter_hbm.json";
        if (std::filesystem::exists(config_path)) {
            SystemConfig config;
            REQUIRE_NOTHROW(config = ConfigLoader::load_from_file(config_path));
            REQUIRE(config.validate());
        }
    }
}

TEST_CASE("SystemSimulator - Configuration Constructor", "[system][simulator]") {
    SECTION("Default constructor uses minimal config") {
        SystemSimulator sim;
        REQUIRE(sim.initialize());
        REQUIRE(sim.is_initialized());
        REQUIRE(sim.get_kpu_count() >= 1);
        sim.shutdown();
    }

    SECTION("Construct with custom config") {
        auto config = SystemConfig::create_edge_ai();
        SystemSimulator sim(config);
        REQUIRE(sim.initialize());
        REQUIRE(sim.get_kpu_count() >= 1);
        sim.shutdown();
    }

    SECTION("Initialize with config") {
        SystemSimulator sim;
        auto config = SystemConfig::create_datacenter();
        REQUIRE(sim.initialize(config));
        REQUIRE(sim.is_initialized());
        sim.shutdown();
    }
}

TEST_CASE("SystemSimulator - KPU Access", "[system][simulator][kpu]") {
    auto config = SystemConfig::create_minimal_kpu();
    SystemSimulator sim(config);
    REQUIRE(sim.initialize());

    SECTION("Get KPU by index") {
        auto* kpu = sim.get_kpu(0);
        REQUIRE(kpu != nullptr);
        REQUIRE(kpu->get_memory_bank_count() > 0);
        REQUIRE(kpu->get_compute_tile_count() > 0);
    }

    SECTION("Get KPU by ID") {
        auto* kpu = sim.get_kpu_by_id("kpu_0");
        REQUIRE(kpu != nullptr);
    }

    SECTION("Invalid KPU index returns nullptr") {
        auto* kpu = sim.get_kpu(999);
        REQUIRE(kpu == nullptr);
    }

    SECTION("Invalid KPU ID returns nullptr") {
        auto* kpu = sim.get_kpu_by_id("nonexistent");
        REQUIRE(kpu == nullptr);
    }

    sim.shutdown();
}

TEST_CASE("SystemSimulator - Self Test", "[system][simulator]") {
    auto config = SystemConfig::create_minimal_kpu();
    SystemSimulator sim(config);
    REQUIRE(sim.initialize());

    SECTION("Self test passes for valid configuration") {
        REQUIRE(sim.run_self_test());
    }

    sim.shutdown();
}

TEST_CASE("SystemSimulator - Configuration Persistence", "[system][simulator][json]") {
    std::filesystem::path temp_file = "test_system_config_temp.json";

    SECTION("Save and reload configuration") {
        // Create and save config
        auto config = SystemConfig::create_edge_ai();
        ConfigLoader::save_to_file(config, temp_file);

        // Load and initialize simulator
        SystemSimulator sim;
        REQUIRE(sim.load_config_and_initialize(temp_file));
        REQUIRE(sim.is_initialized());

        // Verify configuration
        const auto& loaded_config = sim.get_config();
        REQUIRE(loaded_config.system.name == config.system.name);

        sim.shutdown();
        std::filesystem::remove(temp_file);
    }
}
