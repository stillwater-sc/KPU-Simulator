#include "sw/system/system_config.hpp"
#include <sstream>
#include <algorithm>

namespace sw::sim {

//=============================================================================
// AcceleratorConfig Equality Operators
//=============================================================================

bool AcceleratorConfig::operator==(const AcceleratorConfig& other) const {
    if (type != other.type || id != other.id || description != other.description) {
        return false;
    }

    // Compare type-specific configs based on type
    switch (type) {
        case AcceleratorType::KPU:
            return kpu_config.has_value() == other.kpu_config.has_value();
            // Note: We're doing a simple existence check rather than deep comparison
            // Deep comparison would require operator== for all nested config types
        case AcceleratorType::GPU:
            return gpu_config.has_value() == other.gpu_config.has_value();
        case AcceleratorType::TPU:
            return tpu_config.has_value() == other.tpu_config.has_value();
        case AcceleratorType::NPU:
            return npu_config.has_value() == other.npu_config.has_value();
        case AcceleratorType::CGRA:
            return cgra_config.has_value() == other.cgra_config.has_value();
        case AcceleratorType::DSP:
            return dsp_config.has_value() == other.dsp_config.has_value();
        case AcceleratorType::FPGA:
            return fpga_config.has_value() == other.fpga_config.has_value();
        default:
            return true;
    }
}

void InterconnectConfig::clear() {
    host_to_accelerator = HostToAcceleratorConfig{};
    accelerator_to_accelerator = AcceleratorToAcceleratorConfig{};
	on_chip = OnChipConfig{};
	network.reset();
}
bool InterconnectConfig::is_empty() const {
    return host_to_accelerator.type == "None" &&
           accelerator_to_accelerator.type == "None" &&
           !network.has_value();
}

bool InterconnectConfig::operator==(const InterconnectConfig& other) const {
    return (host_to_accelerator.type == other.host_to_accelerator.type) &&
           (accelerator_to_accelerator.type == other.accelerator_to_accelerator.type) &&
           (network.has_value() == other.network.has_value() &&
            (!network.has_value() || (network->type == other.network->type)));
}

//=============================================================================
// SystemConfig State Management Functions
//=============================================================================

void SystemConfig::clear() {
    system = SystemInfo{};
    host = HostConfig{};
    accelerators.clear();
    interconnect = InterconnectConfig{};
    system_services = SystemServicesConfig{};
}

bool SystemConfig::is_empty() const {
    return system.name.empty() && host.memory.modules.empty() && accelerators.empty();
}

bool SystemConfig::operator==(const SystemConfig& other) const {
    return (system.name == other.system.name) &&
           (host.memory.modules == other.host.memory.modules) &&
           (accelerators == other.accelerators) &&
           (interconnect == other.interconnect) &&
           (system_services.memory_manager.enabled == other.system_services.memory_manager.enabled);
}

//=============================================================================
// SystemConfig Validation
//=============================================================================

bool SystemConfig::validate() const {
    return get_validation_errors().empty();
}

std::string SystemConfig::get_validation_errors() const {
    std::stringstream errors;

    // Check required fields
    if (system.name.empty()) {
        errors << "System name is required.\n";
    }

    // Validate host memory
    if (host.memory.modules.empty()) {
        errors << "At least one host memory module is required.\n";
    }

    // Validate memory module capacities
    for (const auto& module : host.memory.modules) {
        if (module.capacity_gb == 0) {
            errors << "Memory module '" << module.id << "' has zero capacity.\n";
        }
        if (module.bandwidth_gbps <= 0.0f) {
            errors << "Memory module '" << module.id << "' has invalid bandwidth.\n";
        }
    }

    // Validate accelerators
    if (accelerators.empty()) {
        errors << "Warning: No accelerators configured (host-only system).\n";
    }

    for (const auto& accel : accelerators) {
        if (accel.id.empty()) {
            errors << "Accelerator must have an ID.\n";
        }

        // Validate KPU configuration
        if (accel.type == AcceleratorType::KPU && accel.kpu_config.has_value()) {
            const auto& kpu = accel.kpu_config.value();

            if (kpu.memory.banks.empty()) {
                errors << "KPU '" << accel.id << "' must have at least one memory bank.\n";
            }

            if (kpu.compute_fabric.tiles.empty()) {
                errors << "KPU '" << accel.id << "' must have at least one compute tile.\n";
            }

            // Check scratchpad count vs compute tiles
            if (kpu.memory.scratchpads.size() < kpu.compute_fabric.tiles.size()) {
                errors << "KPU '" << accel.id << "' should have at least as many scratchpads as compute tiles.\n";
            }

            // Validate DMA engines
            size_t recommended_dma_count = kpu.memory.banks.size() * 2;
            if (kpu.data_movement.dma_engines.size() > recommended_dma_count) {
                errors << "KPU '" << accel.id << "' has more DMA engines than recommended ("
                       << kpu.data_movement.dma_engines.size() << " vs " << recommended_dma_count << ").\n";
            }
        }
        else if (accel.type == AcceleratorType::GPU && accel.gpu_config.has_value()) {
            const auto& gpu = accel.gpu_config.value();
            if (gpu.compute_units == 0) {
                errors << "GPU '" << accel.id << "' must have at least one compute unit.\n";
            }
        }
        else if (accel.type == AcceleratorType::NPU && accel.npu_config.has_value()) {
            const auto& npu = accel.npu_config.value();
            if (npu.tops_int8 == 0 && npu.tops_fp16 == 0) {
                errors << "NPU '" << accel.id << "' must have non-zero TOPS rating.\n";
            }
        }
        else {
            errors << "Accelerator '" << accel.id << "' is missing type-specific configuration.\n";
        }
    }

    // Validate interconnect for multi-accelerator systems
    if (accelerators.size() > 1) {
        if (interconnect.accelerator_to_accelerator.type == "None") {
            errors << "Warning: Multiple accelerators without inter-accelerator interconnect may cause bottlenecks.\n";
        }
    }

    return errors.str();
}

//=============================================================================
// SystemConfig Utility Functions
//=============================================================================

size_t SystemConfig::get_kpu_count() const {
    return std::count_if(accelerators.begin(), accelerators.end(),
        [](const AcceleratorConfig& a) { return a.type == AcceleratorType::KPU; });
}

size_t SystemConfig::get_gpu_count() const {
    return std::count_if(accelerators.begin(), accelerators.end(),
        [](const AcceleratorConfig& a) { return a.type == AcceleratorType::GPU; });
}

size_t SystemConfig::get_tpu_count() const {
    return std::count_if(accelerators.begin(), accelerators.end(),
        [](const AcceleratorConfig& a) { return a.type == AcceleratorType::TPU; });
}

size_t SystemConfig::get_npu_count() const {
    return std::count_if(accelerators.begin(), accelerators.end(),
        [](const AcceleratorConfig& a) { return a.type == AcceleratorType::NPU; });
}

size_t SystemConfig::get_cgra_count() const {
    return std::count_if(accelerators.begin(), accelerators.end(),
        [](const AcceleratorConfig& a) { return a.type == AcceleratorType::CGRA; });
}

size_t SystemConfig::get_dsp_count() const {
    return std::count_if(accelerators.begin(), accelerators.end(),
        [](const AcceleratorConfig& a) { return a.type == AcceleratorType::DSP; });
}

size_t SystemConfig::get_fpga_count() const {
    return std::count_if(accelerators.begin(), accelerators.end(),
        [](const AcceleratorConfig& a) { return a.type == AcceleratorType::FPGA; });
}

const AcceleratorConfig* SystemConfig::find_accelerator(const std::string& id) const {
    auto it = std::find_if(accelerators.begin(), accelerators.end(),
        [&id](const AcceleratorConfig& a) { return a.id == id; });
    return (it != accelerators.end()) ? &(*it) : nullptr;
}

//=============================================================================
// SystemConfig Factory Methods
//=============================================================================

SystemConfig SystemConfig::create_minimal_kpu() {
    SystemConfig config;

    // System info
    config.system.name = "Minimal KPU System";
    config.system.description = "Basic single-KPU system for testing";

    // Host configuration
    config.host.cpu.core_count = 4;
    config.host.cpu.frequency_mhz = 2400;

    MemoryModuleConfig host_mem;
    host_mem.id = "host_mem_0";
    host_mem.type = "DDR4";
    host_mem.form_factor = "DIMM";
    host_mem.capacity_gb = 16;
    host_mem.bandwidth_gbps = 25.6f;
    config.host.memory.modules.push_back(host_mem);

    // KPU accelerator
    AcceleratorConfig kpu_accel;
    kpu_accel.type = AcceleratorType::KPU;
    kpu_accel.id = "kpu_0";
    kpu_accel.description = "Primary knowledge processing unit";

    KPUConfig kpu;
    kpu.memory.type = "GDDR6";
    kpu.memory.form_factor = "PCB";

    // Memory banks
    for (int i = 0; i < 2; ++i) {
        KPUMemoryBankConfig bank;
        bank.id = "bank_" + std::to_string(i);
        bank.capacity_mb = 1024;
        bank.bandwidth_gbps = 100.0f;
        kpu.memory.banks.push_back(bank);
    }

    // L3 tiles
    for (int i = 0; i < 4; ++i) {
        KPUTileConfig tile;
        tile.id = "l3_" + std::to_string(i);
        tile.capacity_kb = 128;
        kpu.memory.l3_tiles.push_back(tile);
    }

    // L2 banks
    for (int i = 0; i < 8; ++i) {
        KPUTileConfig bank;
        bank.id = "l2_" + std::to_string(i);
        bank.capacity_kb = 64;
        kpu.memory.l2_banks.push_back(bank);
    }

    // Scratchpads
    for (int i = 0; i < 2; ++i) {
        KPUScratchpadConfig scratch;
        scratch.id = "scratch_" + std::to_string(i);
        scratch.capacity_kb = 64;
        kpu.memory.scratchpads.push_back(scratch);
    }

    // Compute tiles
    for (int i = 0; i < 2; ++i) {
        ComputeTileConfig tile;
        tile.id = "tile_" + std::to_string(i);
        tile.type = "systolic";
        tile.systolic_rows = 16;
        tile.systolic_cols = 16;
        tile.datatype = "fp32";
        kpu.compute_fabric.tiles.push_back(tile);
    }

    // DMA engines
    for (int i = 0; i < 2; ++i) {
        DMAEngineConfig dma;
        dma.id = "dma_" + std::to_string(i);
        dma.bandwidth_gbps = 50.0f;
        kpu.data_movement.dma_engines.push_back(dma);
    }

    // Block movers
    for (int i = 0; i < 4; ++i) {
        BlockMoverConfig mover;
        mover.id = "block_mover_" + std::to_string(i);
        kpu.data_movement.block_movers.push_back(mover);
    }

    // Streamers
    for (int i = 0; i < 8; ++i) {
        StreamerConfig streamer;
        streamer.id = "streamer_" + std::to_string(i);
        kpu.data_movement.streamers.push_back(streamer);
    }

    kpu_accel.kpu_config = kpu;
    config.accelerators.push_back(kpu_accel);

    // Interconnect
    config.interconnect.host_to_accelerator.type = "PCIe";
    PCIeConfig pcie;
    pcie.generation = 4;
    pcie.lanes = 16;
    pcie.bandwidth_gbps = 32.0f;
    config.interconnect.host_to_accelerator.pcie_config = pcie;

    return config;
}

SystemConfig SystemConfig::create_edge_ai() {
    SystemConfig config;

    config.system.name = "Edge AI System";
    config.system.description = "KPU + NPU heterogeneous system for edge inference";

    // Host with LPDDR5
    config.host.cpu.core_count = 8;
    config.host.cpu.frequency_mhz = 2000;

    MemoryModuleConfig host_mem;
    host_mem.id = "host_mem_0";
    host_mem.type = "LPDDR5";
    host_mem.form_factor = "OnPackage";
    host_mem.capacity_gb = 8;
    host_mem.bandwidth_gbps = 51.2f;
    config.host.memory.modules.push_back(host_mem);

    // Smaller KPU
    AcceleratorConfig kpu_accel;
    kpu_accel.type = AcceleratorType::KPU;
    kpu_accel.id = "kpu_0";

    KPUConfig kpu;
    kpu.memory.type = "LPDDR5";
    kpu.memory.form_factor = "OnPackage";

    KPUMemoryBankConfig bank;
    bank.id = "bank_0";
    bank.capacity_mb = 512;
    bank.bandwidth_gbps = 68.0f;
    kpu.memory.banks.push_back(bank);

    // Scratchpad (at least one per compute tile)
    KPUScratchpadConfig scratch;
    scratch.id = "scratch_0";
    scratch.capacity_kb = 128;
    kpu.memory.scratchpads.push_back(scratch);

    // Compute tile
    ComputeTileConfig tile;
    tile.id = "tile_0";
    tile.systolic_rows = 8;
    tile.systolic_cols = 8;
    tile.datatype = "fp16";
    kpu.compute_fabric.tiles.push_back(tile);

    // DMA engine
    DMAEngineConfig dma;
    dma.id = "dma_0";
    dma.bandwidth_gbps = 34.0f;
    kpu.data_movement.dma_engines.push_back(dma);

    kpu_accel.kpu_config = kpu;
    config.accelerators.push_back(kpu_accel);

    // Set up accelerator-to-accelerator interconnect
    config.interconnect.accelerator_to_accelerator.type = "NoC";
    NoCConfig noc;
    noc.topology = "ring";
    noc.router_count = 2;
    noc.link_bandwidth_gbps = 16.0f;
    config.interconnect.accelerator_to_accelerator.noc_config = noc;

    // NPU
    AcceleratorConfig npu_accel;
    npu_accel.type = AcceleratorType::NPU;
    npu_accel.id = "npu_0";

    NPUConfig npu;
    npu.tops_int8 = 40;
    npu.tops_fp16 = 20;
    npu.memory.type = "OnChip";
    npu.memory.capacity_mb = 16;
    npu_accel.npu_config = npu;
    config.accelerators.push_back(npu_accel);

    return config;
}

SystemConfig SystemConfig::create_datacenter() {
    SystemConfig config;

    config.system.name = "Datacenter AI Cluster Node";
    config.system.description = "High-performance datacenter system with HBM";

    // High-end host
    config.host.cpu.core_count = 64;
    config.host.cpu.frequency_mhz = 3500;
    config.host.cpu.cache_l3_kb = 262144;

    // Multiple DDR5 channels (reduced for CI/testing)
    for (int i = 0; i < 4; ++i) {
        MemoryModuleConfig mem;
        mem.id = "host_mem_" + std::to_string(i);
        mem.type = "DDR5";
        mem.form_factor = "DIMM";
        mem.capacity_gb = 4;  // Reduced from 128 GB to 4 GB for CI
        mem.bandwidth_gbps = 89.6f;
        config.host.memory.modules.push_back(mem);
    }

    // High-end KPU with HBM
    AcceleratorConfig kpu_accel;
    kpu_accel.type = AcceleratorType::KPU;
    kpu_accel.id = "kpu_0";

    KPUConfig kpu;
    kpu.memory.type = "HBM3";
    kpu.memory.form_factor = "Interposer";

    // Multiple HBM stacks (reduced for CI/testing)
    for (int i = 0; i < 4; ++i) {
        KPUMemoryBankConfig bank;
        bank.id = "bank_" + std::to_string(i);
        bank.capacity_mb = 256;  // Reduced from 8192 MB to 256 MB for CI
        bank.bandwidth_gbps = 819.0f;
        bank.latency_ns = 10;
        kpu.memory.banks.push_back(bank);
    }

    // Scratchpads (at least one per compute tile)
    for (int i = 0; i < 4; ++i) {
        KPUScratchpadConfig scratch;
        scratch.id = "scratch_" + std::to_string(i);
        scratch.capacity_kb = 256;
        kpu.memory.scratchpads.push_back(scratch);
    }

    // Large compute fabric
    for (int i = 0; i < 4; ++i) {
        ComputeTileConfig tile;
        tile.id = "tile_" + std::to_string(i);
        tile.systolic_rows = 32;
        tile.systolic_cols = 32;
        kpu.compute_fabric.tiles.push_back(tile);
    }

    // DMA engines
    for (int i = 0; i < 4; ++i) {
        DMAEngineConfig dma;
        dma.id = "dma_" + std::to_string(i);
        dma.bandwidth_gbps = 200.0f;
        kpu.data_movement.dma_engines.push_back(dma);
    }

    kpu_accel.kpu_config = kpu;
    config.accelerators.push_back(kpu_accel);

    // GPU
    AcceleratorConfig gpu_accel;
    gpu_accel.type = AcceleratorType::GPU;
    gpu_accel.id = "gpu_0";

    GPUConfig gpu;
    gpu.compute_units = 128;
    gpu.clock_mhz = 2100;
    gpu.memory.type = "HBM3";
    gpu.memory.capacity_gb = 2;  // Reduced from 80 GB to 2 GB for CI
    gpu.memory.bandwidth_gbps = 3200.0f;
    gpu_accel.gpu_config = gpu;
    config.accelerators.push_back(gpu_accel);

    // High-speed interconnect
    config.interconnect.host_to_accelerator.type = "PCIe";
    PCIeConfig pcie;
    pcie.generation = 5;
    pcie.lanes = 16;
    pcie.bandwidth_gbps = 128.0f;
    config.interconnect.host_to_accelerator.pcie_config = pcie;

    // Accelerator-to-accelerator interconnect
    config.interconnect.accelerator_to_accelerator.type = "NVLink";
    NoCConfig noc;
    noc.topology = "crossbar";
    noc.router_count = 2;
    noc.link_bandwidth_gbps = 900.0f;
    config.interconnect.accelerator_to_accelerator.noc_config = noc;

    return config;
}

} // namespace sw::sim
