#include "sw/system/toplevel.hpp"
#include "sw/system/config_loader.hpp"
#include "sw/kpu/kpu_simulator.hpp"
#include <iostream>
#include <iomanip>

namespace sw::sim {

//=============================================================================
// Constructors and Destructor
//=============================================================================

SystemSimulator::SystemSimulator()
    : config_(SystemConfig::create_minimal_kpu()) {
    std::cout << "[SystemSimulator] Constructor called with default config\n";
}

SystemSimulator::SystemSimulator(const SystemConfig& config)
    : config_(config) {
    std::cout << "[SystemSimulator] Constructor called with custom config\n";
}

SystemSimulator::SystemSimulator(const std::filesystem::path& config_file) {
    std::cout << "[SystemSimulator] Loading configuration from: " << config_file << "\n";
    config_ = ConfigLoader::load_from_file(config_file);
}

SystemSimulator::~SystemSimulator() {
    if (initialized_) {
        shutdown();
    }
}

SystemSimulator::SystemSimulator(SystemSimulator&&) noexcept = default;
SystemSimulator& SystemSimulator::operator=(SystemSimulator&&) noexcept = default;

//=============================================================================
// Initialization and Shutdown
//=============================================================================

bool SystemSimulator::initialize() {
    if (initialized_) {
        std::cout << "[SystemSimulator] Already initialized\n";
        return true;
    }

    std::cout << "[SystemSimulator] Initializing system: " << config_.system.name << "\n";

    // Validate configuration
    if (!config_.validate()) {
        std::cerr << "[SystemSimulator] Configuration validation failed:\n";
        std::cerr << config_.get_validation_errors();
        return false;
    }

    // Create components based on configuration
    try {
        create_components_from_config();
    }
    catch (const std::exception& e) {
        std::cerr << "[SystemSimulator] Failed to create components: " << e.what() << "\n";
        destroy_components();
        return false;
    }

    initialized_ = true;
    std::cout << "[SystemSimulator] Initialization complete\n";
    print_config();
    return true;
}

bool SystemSimulator::initialize(const SystemConfig& config) {
    if (initialized_) {
        shutdown();
    }
    config_ = config;
    return initialize();
}

bool SystemSimulator::load_config_and_initialize(const std::filesystem::path& config_file) {
    if (initialized_) {
        shutdown();
    }

    try {
        config_ = ConfigLoader::load_from_file(config_file);
        return initialize();
    }
    catch (const std::exception& e) {
        std::cerr << "[SystemSimulator] Failed to load config: " << e.what() << "\n";
        return false;
    }
}

void SystemSimulator::shutdown() {
    if (!initialized_) {
        std::cout << "[SystemSimulator] Already shut down\n";
        return;
    }

    std::cout << "[SystemSimulator] Shutting down system components...\n";

    destroy_components();

    initialized_ = false;
    std::cout << "[SystemSimulator] Shutdown complete\n";
}

//=============================================================================
// Component Management
//=============================================================================

void SystemSimulator::create_components_from_config() {
    std::cout << "[SystemSimulator] Creating components from configuration...\n";

    // Create KPU instances
    for (const auto& accel_config : config_.accelerators) {
        if (accel_config.type == AcceleratorType::KPU && accel_config.kpu_config.has_value()) {
            std::cout << "[SystemSimulator] Creating KPU: " << accel_config.id << "\n";
            auto* kpu = create_kpu_from_config(accel_config.kpu_config.value());
            if (kpu) {
                kpu_instances_.emplace_back(kpu);
            }
        }
        else if (accel_config.type == AcceleratorType::GPU) {
            std::cout << "[SystemSimulator] GPU support not yet implemented: " << accel_config.id << "\n";
        }
        else if (accel_config.type == AcceleratorType::NPU) {
            std::cout << "[SystemSimulator] NPU support not yet implemented: " << accel_config.id << "\n";
        }
    }

    std::cout << "[SystemSimulator] Created " << kpu_instances_.size() << " KPU instance(s)\n";
}

void SystemSimulator::destroy_components() {
    std::cout << "[SystemSimulator] Destroying components...\n";
    kpu_instances_.clear();
}

sw::kpu::KPUSimulator* SystemSimulator::create_kpu_from_config(const KPUConfig& kpu_config) {
    // Convert KPUConfig to sw::kpu::KPUSimulator::Config
    sw::kpu::KPUSimulator::Config sim_config;

    // Memory configuration
    sim_config.memory_bank_count = kpu_config.memory.banks.size();
    if (!kpu_config.memory.banks.empty()) {
        sim_config.memory_bank_capacity_mb = kpu_config.memory.banks[0].capacity_mb;
        sim_config.memory_bandwidth_gbps = kpu_config.memory.banks[0].bandwidth_gbps;
    }

    // Scratchpad configuration
    sim_config.scratchpad_count = kpu_config.memory.scratchpads.size();
    if (!kpu_config.memory.scratchpads.empty()) {
        sim_config.scratchpad_capacity_kb = kpu_config.memory.scratchpads[0].capacity_kb;
    }

    // L3 and L2 configuration
    sim_config.l3_tile_count = kpu_config.memory.l3_tiles.size();
    if (!kpu_config.memory.l3_tiles.empty()) {
        sim_config.l3_tile_capacity_kb = kpu_config.memory.l3_tiles[0].capacity_kb;
    }

    sim_config.l2_bank_count = kpu_config.memory.l2_banks.size();
    if (!kpu_config.memory.l2_banks.empty()) {
        sim_config.l2_bank_capacity_kb = kpu_config.memory.l2_banks[0].capacity_kb;
    }

    // Compute configuration
    sim_config.compute_tile_count = kpu_config.compute_fabric.tiles.size();
    if (!kpu_config.compute_fabric.tiles.empty()) {
        const auto& tile = kpu_config.compute_fabric.tiles[0];
        sim_config.use_systolic_arrays = (tile.type == "systolic");
        sim_config.systolic_array_rows = tile.systolic_rows;
        sim_config.systolic_array_cols = tile.systolic_cols;
    }

    // Data movement configuration
    sim_config.dma_engine_count = kpu_config.data_movement.dma_engines.size();
    sim_config.block_mover_count = kpu_config.data_movement.block_movers.size();
    sim_config.streamer_count = kpu_config.data_movement.streamers.size();

    return new sw::kpu::KPUSimulator(sim_config);
}

//=============================================================================
// Component Access
//=============================================================================

size_t SystemSimulator::get_kpu_count() const {
    return kpu_instances_.size();
}

sw::kpu::KPUSimulator* SystemSimulator::get_kpu(size_t index) {
    if (index >= kpu_instances_.size()) {
        return nullptr;
    }
    return kpu_instances_[index].get();
}

sw::kpu::KPUSimulator* SystemSimulator::get_kpu_by_id(const std::string& id) {
    size_t idx = 0;
    for (const auto& accel_config : config_.accelerators) {
        if (accel_config.type == AcceleratorType::KPU) {
            if (accel_config.id == id) {
                return get_kpu(idx);
            }
            idx++;
        }
    }
    return nullptr;
}

//=============================================================================
// Testing and Status
//=============================================================================

bool SystemSimulator::run_self_test() {
    if (!initialized_) {
        std::cout << "[SystemSimulator] Cannot run self test - not initialized\n";
        return false;
    }

    std::cout << "[SystemSimulator] Running self test...\n";

    bool test_passed = true;

    // Test KPU instances
    for (size_t i = 0; i < kpu_instances_.size(); ++i) {
        std::cout << "[SystemSimulator] Testing KPU " << i << "...\n";
        auto* kpu = kpu_instances_[i].get();

        // Simple validation
        if (kpu->get_memory_bank_count() == 0) {
            std::cout << "[SystemSimulator] KPU " << i << " has no memory banks!\n";
            test_passed = false;
        }
        if (kpu->get_compute_tile_count() == 0) {
            std::cout << "[SystemSimulator] KPU " << i << " has no compute tiles!\n";
            test_passed = false;
        }
    }

    std::cout << "[SystemSimulator] Self test "
              << (test_passed ? "PASSED" : "FAILED") << "\n";
    return test_passed;
}

void SystemSimulator::print_config() const {
    std::cout << "\n========================================\n";
    std::cout << "System Configuration: " << config_.system.name << "\n";
    std::cout << "========================================\n";

    // Host
    std::cout << "\nHost CPU:\n";
    std::cout << "  Cores: " << config_.host.cpu.core_count << " @ "
              << config_.host.cpu.frequency_mhz << " MHz\n";
    std::cout << "  Cache: L1=" << config_.host.cpu.cache_l1_kb << "KB, "
              << "L2=" << config_.host.cpu.cache_l2_kb << "KB, "
              << "L3=" << config_.host.cpu.cache_l3_kb << "KB\n";

    std::cout << "\nHost Memory:\n";
    for (const auto& mem : config_.host.memory.modules) {
        std::cout << "  " << mem.id << ": " << mem.capacity_gb << "GB "
                  << mem.type << " @ " << mem.bandwidth_gbps << " GB/s\n";
    }

    // Accelerators
    std::cout << "\nAccelerators: " << config_.accelerators.size() << "\n";
    for (const auto& accel : config_.accelerators) {
        std::cout << "  " << accel.id << " (";
        switch (accel.type) {
            case AcceleratorType::KPU: std::cout << "KPU"; break;
            case AcceleratorType::GPU: std::cout << "GPU"; break;
            case AcceleratorType::NPU: std::cout << "NPU"; break;
            default: std::cout << "Unknown"; break;
        }
        std::cout << ")\n";

        if (accel.kpu_config.has_value()) {
            const auto& kpu = accel.kpu_config.value();
            std::cout << "    Memory: " << kpu.memory.banks.size() << " banks, "
                      << kpu.memory.type << "\n";
            std::cout << "    Compute: " << kpu.compute_fabric.tiles.size() << " tiles\n";
        }
    }

    // Interconnect
    std::cout << "\nInterconnect:\n";
    std::cout << "  Host-to-Accelerator: " << config_.interconnect.host_to_accelerator.type << "\n";
    if (config_.interconnect.host_to_accelerator.pcie_config.has_value()) {
        const auto& pcie = config_.interconnect.host_to_accelerator.pcie_config.value();
        std::cout << "    PCIe Gen" << pcie.generation << " x" << pcie.lanes
                  << " (" << pcie.bandwidth_gbps << " GB/s)\n";
    }

    std::cout << "========================================\n\n";
}

void SystemSimulator::print_status() const {
    std::cout << "\n========================================\n";
    std::cout << "System Status\n";
    std::cout << "========================================\n";
    std::cout << "Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
    std::cout << "KPU Instances: " << kpu_instances_.size() << "\n";
    std::cout << "========================================\n\n";
}

} // namespace sw::sim