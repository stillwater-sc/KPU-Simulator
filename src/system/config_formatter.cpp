#include "sw/system/config_formatter.hpp"
#include <sstream>
#include <iomanip>

namespace sw::sim {

//=============================================================================
// Helper Functions
//=============================================================================

namespace {
    // Format size in human-readable form
    std::string format_size(uint32_t value, const std::string& unit) {
        std::ostringstream oss;
        oss << value << " " << unit;
        return oss.str();
    }

    // Format bandwidth
    std::string format_bandwidth(float gbps) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << gbps << " GB/s";
        return oss.str();
    }
}

//=============================================================================
// System Info and Top-Level Config
//=============================================================================

std::ostream& operator<<(std::ostream& os, const SystemInfo& info) {
    os << "System: " << info.name << "\n";
    if (!info.description.empty()) {
        os << "  Description: " << info.description << "\n";
    }
    if (info.clock_frequency_mhz.has_value()) {
        os << "  Clock: " << info.clock_frequency_mhz.value() << " MHz\n";
    }
    return os;
}

//=============================================================================
// Host Configuration
//=============================================================================

std::ostream& operator<<(std::ostream& os, const CPUConfig& config) {
    os << "CPU:\n";
    os << "  Cores: " << config.core_count << " @ " << config.frequency_mhz << " MHz\n";
    os << "  Cache: L1=" << config.cache_l1_kb << "KB, "
       << "L2=" << config.cache_l2_kb << "KB, "
       << "L3=" << config.cache_l3_kb << "KB\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const MemoryModuleConfig& config) {
    os << "  [" << config.id << "] " << config.capacity_gb << "GB " << config.type
       << " (" << config.form_factor << ") @ " << format_bandwidth(config.bandwidth_gbps)
       << ", " << config.latency_ns << "ns, " << config.channels << " channels\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const HostMemoryConfig& config) {
    os << "Host Memory:\n";
    os << "  DRAM Controller: " << config.dram_controller.channel_count << " channels, "
       << config.dram_controller.data_width_bits << "-bit\n";
    os << "  Modules:\n";
    for (const auto& module : config.modules) {
        os << "    " << module;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const StorageConfig& config) {
    os << "  [" << config.id << "] " << config.capacity_gb << "GB " << config.type
       << " (R:" << config.read_bandwidth_mbps << "MB/s, W:"
       << config.write_bandwidth_mbps << "MB/s, " << config.latency_us << "us)\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const HostConfig& config) {
    os << config.cpu;
    os << config.memory;
    if (!config.storage.empty()) {
        os << "Storage:\n";
        for (const auto& storage : config.storage) {
            os << "  " << storage;
        }
    }
    return os;
}

//=============================================================================
// KPU Memory Configuration
//=============================================================================

std::ostream& operator<<(std::ostream& os, const KPUMemoryBankConfig& config) {
    os << "    [" << config.id << "] " << format_size(config.capacity_mb, "MB")
       << " @ " << format_bandwidth(config.bandwidth_gbps)
       << ", " << config.latency_ns << "ns\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const KPUTileConfig& config) {
    os << "    [" << config.id << "] " << format_size(config.capacity_kb, "KB") << "\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const KPUL1Config& config) {
    os << "    [" << config.id << "] " << format_size(config.capacity_kb, "KB") << "\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const KPUScratchpadConfig& config) {
    os << "    [" << config.id << "] " << format_size(config.capacity_kb, "KB") << "\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const KPUMemoryConfig& config) {
    os << "  Memory Type: " << config.type << " (" << config.form_factor << ")\n";

    if (!config.banks.empty()) {
        os << "  External Memory Banks (" << config.banks.size() << "):\n";
        for (const auto& bank : config.banks) {
            os << bank;
        }
    }

    if (!config.l3_tiles.empty()) {
        os << "  L3 Tiles (" << config.l3_tiles.size() << "):\n";
        for (const auto& tile : config.l3_tiles) {
            os << tile;
        }
    }

    if (!config.l2_banks.empty()) {
        os << "  L2 Banks (" << config.l2_banks.size() << "):\n";
        for (const auto& bank : config.l2_banks) {
            os << bank;
        }
    }

    if (!config.l1_buffers.empty()) {
        os << "  L1 Buffers (" << config.l1_buffers.size() << "):\n";
        for (const auto& buffer : config.l1_buffers) {
            os << buffer;
        }
    }

    if (!config.scratchpads.empty()) {
        os << "  Scratchpads (" << config.scratchpads.size() << "):\n";
        for (const auto& scratch : config.scratchpads) {
            os << scratch;
        }
    }

    return os;
}

//=============================================================================
// KPU Compute Configuration
//=============================================================================

std::ostream& operator<<(std::ostream& os, const ComputeTileConfig& config) {
    os << "    [" << config.id << "] " << config.type;
    if (config.type == "systolic") {
        os << " (" << config.systolic_rows << "x" << config.systolic_cols << ")";
    }
    os << ", " << config.datatype << "\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const KPUComputeConfig& config) {
    os << "  Compute Tiles (" << config.tiles.size() << "):\n";
    for (const auto& tile : config.tiles) {
        os << tile;
    }
    return os;
}

//=============================================================================
// KPU Data Movement Configuration
//=============================================================================

std::ostream& operator<<(std::ostream& os, const DMAEngineConfig& config) {
    os << "    [" << config.id << "] " << format_bandwidth(config.bandwidth_gbps)
       << ", " << config.channels << " channel(s)\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const BlockMoverConfig& config) {
    os << "    [" << config.id << "]\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const StreamerConfig& config) {
    os << "    [" << config.id << "]\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const KPUDataMovementConfig& config) {
    if (!config.dma_engines.empty()) {
        os << "  DMA Engines (" << config.dma_engines.size() << "):\n";
        for (const auto& dma : config.dma_engines) {
            os << dma;
        }
    }

    if (!config.block_movers.empty()) {
        os << "  Block Movers (" << config.block_movers.size() << "):\n";
        for (const auto& mover : config.block_movers) {
            os << mover;
        }
    }

    if (!config.streamers.empty()) {
        os << "  Streamers (" << config.streamers.size() << "):\n";
        for (const auto& streamer : config.streamers) {
            os << streamer;
        }
    }

    return os;
}

//=============================================================================
// Complete KPU Configuration
//=============================================================================

std::ostream& operator<<(std::ostream& os, const KPUConfig& config) {
    os << config.memory;
    os << config.compute_fabric;
    os << config.data_movement;
    return os;
}

//=============================================================================
// Accelerator Configuration
//=============================================================================

std::ostream& operator<<(std::ostream& os, AcceleratorType type) {
    switch (type) {
        case AcceleratorType::KPU:  os << "KPU"; break;
        case AcceleratorType::GPU:  os << "GPU"; break;
        case AcceleratorType::TPU:  os << "TPU"; break;
        case AcceleratorType::NPU:  os << "NPU"; break;
        case AcceleratorType::CGRA: os << "CGRA"; break;
        case AcceleratorType::DSP:  os << "DSP"; break;
        case AcceleratorType::FPGA: os << "FPGA"; break;
        default: os << "Unknown"; break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const AcceleratorConfig& config) {
    os << "Accelerator [" << config.id << "] - " << config.type;
    if (!config.description.empty()) {
        os << " (" << config.description << ")";
    }
    os << "\n";

    if (config.kpu_config.has_value()) {
        os << config.kpu_config.value();
    } else if (config.gpu_config.has_value()) {
        os << "  GPU configuration (details not implemented)\n";
    } else if (config.npu_config.has_value()) {
        os << "  NPU configuration (details not implemented)\n";
    }

    return os;
}

//=============================================================================
// Interconnect Configuration
//=============================================================================

std::ostream& operator<<(std::ostream& os, const PCIeConfig& config) {
    os << "  PCIe Gen" << config.generation << " x" << config.lanes
       << " (" << format_bandwidth(config.bandwidth_gbps) << ")\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const CXLConfig& config) {
    os << "  CXL " << config.version << " (" << format_bandwidth(config.bandwidth_gbps) << ")\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const HostToAcceleratorConfig& config) {
    os << "Host-to-Accelerator: " << config.type << "\n";
    if (config.pcie_config.has_value()) {
        os << config.pcie_config.value();
    }
    if (config.cxl_config.has_value()) {
        os << config.cxl_config.value();
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const NoCConfig& config) {
    os << "  NoC: " << config.topology << ", " << config.router_count << " routers, "
       << format_bandwidth(config.link_bandwidth_gbps) << " per link\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const AcceleratorToAcceleratorConfig& config) {
    os << "Accelerator-to-Accelerator: " << config.type << "\n";
    if (config.noc_config.has_value()) {
        os << config.noc_config.value();
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const AMBAConfig& config) {
    os << "  AMBA: " << config.protocol << "\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const OnChipConfig& config) {
    os << "On-Chip: " << config.type << "\n";
    if (config.amba_config.has_value()) {
        os << config.amba_config.value();
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const InterconnectConfig& config) {
    os << "Interconnect:\n";
    os << config.host_to_accelerator;
    os << config.accelerator_to_accelerator;
    os << config.on_chip;
    if (config.network.has_value()) {
        const auto& net = config.network.value();
        if (net.enabled) {
            os << "Network: " << net.type << " @ " << net.speed_gbps << " Gbps\n";
        }
    }
    return os;
}

//=============================================================================
// System Services
//=============================================================================

std::ostream& operator<<(std::ostream& os, const MemoryManagerConfig& config) {
    os << "  Memory Manager: " << (config.enabled ? "Enabled" : "Disabled");
    if (config.enabled) {
        os << " (Pool: " << format_size(config.pool_size_mb, "MB")
           << ", Alignment: " << config.alignment_bytes << " bytes)";
    }
    os << "\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const SystemServicesConfig& config) {
    os << "System Services:\n";
    os << config.memory_manager;
    os << "  Interrupt Controller: "
       << (config.interrupt_controller.enabled ? "Enabled" : "Disabled") << "\n";
    os << "  Power Management: "
       << (config.power_management.enabled ? "Enabled" : "Disabled") << "\n";
    return os;
}

//=============================================================================
// Complete System Configuration
//=============================================================================

std::ostream& operator<<(std::ostream& os, const SystemConfig& config) {
    os << "========================================\n";
    os << config.system;
    os << "========================================\n\n";

    os << config.host << "\n";

    if (!config.accelerators.empty()) {
        os << "Accelerators (" << config.accelerators.size() << "):\n";
        for (const auto& accel : config.accelerators) {
            os << "\n" << accel;
        }
        os << "\n";
    }

    os << config.interconnect << "\n";
    os << config.system_services;

    os << "========================================\n";

    return os;
}

//=============================================================================
// Formatted Output Functions
//=============================================================================

void print_config(std::ostream& os, const SystemConfig& config, FormatDetail /* detail */) {
    // For now, all detail levels use the same format
    // Future enhancement: implement different formatting based on detail level
    os << config;
}

void print_kpu_config(std::ostream& os, const KPUConfig& config,
                      FormatDetail /* detail */, const std::string& /* indent */) {
    // Future enhancement: implement indentation and detail level control
    os << config;
}

std::string to_string(const SystemConfig& config, FormatDetail detail) {
    std::ostringstream oss;
    print_config(oss, config, detail);
    return oss.str();
}

} // namespace sw::sim
