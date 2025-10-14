#pragma once

#include <sw/system/system_config.hpp>
#include <ostream>
#include <iomanip>

namespace sw::sim {

/**
 * @brief Detail level for configuration formatting
 */
enum class FormatDetail {
    Summary,   ///< Just counts and high-level info
    Standard,  ///< IDs and key specifications (default)
    Full       ///< Complete details including all technical specs
};

//=============================================================================
// Stream Operators - Enable usage like: std::cout << config;
//=============================================================================

// Top-level system configuration
std::ostream& operator<<(std::ostream& os, const SystemConfig& config);
std::ostream& operator<<(std::ostream& os, const SystemInfo& info);

// Host configuration
std::ostream& operator<<(std::ostream& os, const CPUConfig& config);
std::ostream& operator<<(std::ostream& os, const MemoryModuleConfig& config);
std::ostream& operator<<(std::ostream& os, const HostMemoryConfig& config);
std::ostream& operator<<(std::ostream& os, const StorageConfig& config);
std::ostream& operator<<(std::ostream& os, const HostConfig& config);

// KPU accelerator configuration
std::ostream& operator<<(std::ostream& os, const KPUMemoryBankConfig& config);
std::ostream& operator<<(std::ostream& os, const KPUTileConfig& config);
std::ostream& operator<<(std::ostream& os, const KPUL1Config& config);
std::ostream& operator<<(std::ostream& os, const KPUScratchpadConfig& config);
std::ostream& operator<<(std::ostream& os, const KPUMemoryConfig& config);
std::ostream& operator<<(std::ostream& os, const ComputeTileConfig& config);
std::ostream& operator<<(std::ostream& os, const KPUComputeConfig& config);
std::ostream& operator<<(std::ostream& os, const DMAEngineConfig& config);
std::ostream& operator<<(std::ostream& os, const BlockMoverConfig& config);
std::ostream& operator<<(std::ostream& os, const StreamerConfig& config);
std::ostream& operator<<(std::ostream& os, const KPUDataMovementConfig& config);
std::ostream& operator<<(std::ostream& os, const KPUConfig& config);

// Accelerator types
std::ostream& operator<<(std::ostream& os, AcceleratorType type);
std::ostream& operator<<(std::ostream& os, const AcceleratorConfig& config);

// Interconnect configuration
std::ostream& operator<<(std::ostream& os, const PCIeConfig& config);
std::ostream& operator<<(std::ostream& os, const CXLConfig& config);
std::ostream& operator<<(std::ostream& os, const HostToAcceleratorConfig& config);
std::ostream& operator<<(std::ostream& os, const NoCConfig& config);
std::ostream& operator<<(std::ostream& os, const AcceleratorToAcceleratorConfig& config);
std::ostream& operator<<(std::ostream& os, const AMBAConfig& config);
std::ostream& operator<<(std::ostream& os, const OnChipConfig& config);
std::ostream& operator<<(std::ostream& os, const InterconnectConfig& config);

// System services
std::ostream& operator<<(std::ostream& os, const MemoryManagerConfig& config);
std::ostream& operator<<(std::ostream& os, const SystemServicesConfig& config);

//=============================================================================
// Formatted Output Functions - More control over detail level
//=============================================================================

/**
 * @brief Print system configuration with specified detail level
 * @param os Output stream
 * @param config Configuration to print
 * @param detail Level of detail to include
 */
void print_config(std::ostream& os, const SystemConfig& config,
                  FormatDetail detail = FormatDetail::Standard);

/**
 * @brief Print KPU configuration with specified detail level
 * @param os Output stream
 * @param config KPU configuration to print
 * @param detail Level of detail to include
 * @param indent Indentation string (for nested formatting)
 */
void print_kpu_config(std::ostream& os, const KPUConfig& config,
                      FormatDetail detail = FormatDetail::Standard,
                      const std::string& indent = "");

/**
 * @brief Convert configuration to formatted string
 * @param config Configuration to format
 * @param detail Level of detail to include
 * @return Formatted string representation
 */
std::string to_string(const SystemConfig& config,
                      FormatDetail detail = FormatDetail::Standard);

} // namespace sw::sim
