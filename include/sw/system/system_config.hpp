#pragma once

#include <string>
#include <vector>
#include <optional>
#include <cstdint>

namespace sw::sim {

// Forward declarations
struct MemoryModuleConfig;
struct StorageConfig;
struct KPUMemoryConfig;
struct KPUComputeConfig;
struct KPUDataMovementConfig;
struct AcceleratorConfig;
struct InterconnectConfig;
struct SystemServicesConfig;

//=============================================================================
// Host Configuration Structures
//=============================================================================

struct CPUConfig {
    uint32_t core_count{4};
    uint32_t frequency_mhz{2400};
    uint32_t cache_l1_kb{32};
    uint32_t cache_l2_kb{256};
    uint32_t cache_l3_kb{8192};
};

struct DRAMControllerConfig {
    uint32_t channel_count{2};
    uint32_t data_width_bits{64};
};

struct MemoryModuleConfig {
    std::string id;
    std::string type;  // DDR4, DDR5, LPDDR4, LPDDR5
    std::string form_factor;  // DIMM, SODIMM, LPDIMM, OnPackage
    uint32_t capacity_gb{16};
    uint32_t frequency_mhz{3200};
    float bandwidth_gbps{25.6f};
    uint32_t latency_ns{80};
    uint32_t channels{2};
};

struct HostMemoryConfig {
    DRAMControllerConfig dram_controller;
    std::vector<MemoryModuleConfig> modules;
};

struct StorageConfig {
    std::string id;
    std::string type;  // SSD, NVME, HDD
    uint32_t capacity_gb{256};
    uint32_t read_bandwidth_mbps{3500};
    uint32_t write_bandwidth_mbps{3000};
    uint32_t latency_us{100};
};

struct HostConfig {
    CPUConfig cpu;
    HostMemoryConfig memory;
    std::vector<StorageConfig> storage;
};

//=============================================================================
// KPU Accelerator Configuration Structures
//=============================================================================

struct KPUMemoryBankConfig {
    std::string id;
    uint32_t capacity_mb{1024};
    float bandwidth_gbps{100.0f};
    uint32_t latency_ns{20};
};

struct KPUTileConfig {
    std::string id;
    uint32_t capacity_kb{128};
};

struct KPUScratchpadConfig {
    std::string id;
    uint32_t capacity_kb{64};
};

struct KPUMemoryConfig {
    std::string type;  // GDDR6, HBM2, HBM3, LPDDR5, Custom
    std::string form_factor;  // Substrate, PCB, Interposer, 3DStack, OnPackage
    std::vector<KPUMemoryBankConfig> banks;
    std::vector<KPUTileConfig> l3_tiles;
    std::vector<KPUTileConfig> l2_banks;
    std::vector<KPUScratchpadConfig> scratchpads;
};

struct ComputeTileConfig {
    std::string id;
    std::string type{  "systolic"};  // systolic, vector, scalar
    uint32_t systolic_rows{16};
    uint32_t systolic_cols{16};
    std::string datatype{"fp32"};  // fp32, fp16, int8, bfloat16
};

struct KPUComputeConfig {
    std::vector<ComputeTileConfig> tiles;
};

struct DMAEngineConfig {
    std::string id;
    float bandwidth_gbps{50.0f};
    uint32_t channels{1};
};

struct BlockMoverConfig {
    std::string id;
};

struct StreamerConfig {
    std::string id;
};

struct KPUDataMovementConfig {
    std::vector<DMAEngineConfig> dma_engines;
    std::vector<BlockMoverConfig> block_movers;
    std::vector<StreamerConfig> streamers;
};

struct KPUConfig {
    KPUMemoryConfig memory;
    KPUComputeConfig compute_fabric;
    KPUDataMovementConfig data_movement;
};

//=============================================================================
// GPU Accelerator Configuration Structures
//=============================================================================

struct GPUMemoryConfig {
    std::string type;  // GDDR6, GDDR6X, HBM2, HBM2E, HBM3
    std::string form_factor;  // PCB, Substrate, Interposer
    uint32_t capacity_gb{8};
    float bandwidth_gbps{448.0f};
    uint32_t bus_width_bits{256};
};

struct GPUConfig {
    uint32_t compute_units{64};
    uint32_t clock_mhz{1800};
    GPUMemoryConfig memory;
};

//=============================================================================
// NPU Accelerator Configuration Structures
//=============================================================================

struct NPUMemoryConfig {
    std::string type;  // LPDDR5, HBM, OnChip
    uint32_t capacity_mb{16};
    float bandwidth_gbps{200.0f};
};

struct NPUConfig {
    uint32_t tops_int8{40};
    uint32_t tops_fp16{20};
    NPUMemoryConfig memory;
};

//=============================================================================
// Unified Accelerator Configuration
//=============================================================================

enum class AcceleratorType {
    KPU,
    GPU,
    NPU,
    DSP,
    FPGA
};

struct AcceleratorConfig {
    AcceleratorType type{AcceleratorType::KPU};
    std::string id;
    std::string description;

    // Only one of these will be populated based on type
    std::optional<KPUConfig> kpu_config;
    std::optional<GPUConfig> gpu_config;
    std::optional<NPUConfig> npu_config;
};

//=============================================================================
// Interconnect Configuration Structures
//=============================================================================

struct PCIeConfig {
    uint32_t generation{4};
    uint32_t lanes{16};
    float bandwidth_gbps{32.0f};
};

struct CXLConfig {
    std::string version{"2.0"};  // 1.0, 2.0, 3.0
    float bandwidth_gbps{64.0f};
};

struct HostToAcceleratorConfig {
    std::string type{"PCIe"};  // PCIe, CXL, NVLink, CustomFabric
    std::optional<PCIeConfig> pcie_config;
    std::optional<CXLConfig> cxl_config;
};

struct NoCConfig {
    std::string topology{"mesh"};  // mesh, torus, ring, crossbar
    uint32_t router_count{4};
    float link_bandwidth_gbps{16.0f};
};

struct AcceleratorToAcceleratorConfig {
    std::string type{"None"};  // NVLink, InfinityFabric, NoC, None
    std::optional<NoCConfig> noc_config;
};

struct AMBAConfig {
    std::string protocol{"AXI4"};  // AXI4, AXI5, ACE, CHI
};

struct OnChipConfig {
    std::string type{"AMBA"};  // AMBA, CHI, TileLink, Custom
    std::optional<AMBAConfig> amba_config;
};

struct NetworkConfig {
    bool enabled{false};
    std::string type{"Ethernet"};  // Ethernet, RoCE, InfiniBand
    uint32_t speed_gbps{100};
};

struct InterconnectConfig {
    HostToAcceleratorConfig host_to_accelerator;
    AcceleratorToAcceleratorConfig accelerator_to_accelerator;
    OnChipConfig on_chip;
    std::optional<NetworkConfig> network;
};

//=============================================================================
// System Services Configuration
//=============================================================================

struct MemoryManagerConfig {
    bool enabled{true};
    uint32_t pool_size_mb{512};
    uint32_t alignment_bytes{64};
};

struct InterruptControllerConfig {
    bool enabled{true};
};

struct PowerManagementConfig {
    bool enabled{false};
};

struct SystemServicesConfig {
    MemoryManagerConfig memory_manager;
    InterruptControllerConfig interrupt_controller;
    PowerManagementConfig power_management;
};

//=============================================================================
// Top-Level System Configuration
//=============================================================================

struct SystemInfo {
    std::string name{"Unnamed System"};
    std::string description;
    std::optional<uint32_t> clock_frequency_mhz;
};

struct SystemConfig {
    SystemInfo system;
    HostConfig host;
    std::vector<AcceleratorConfig> accelerators;
    InterconnectConfig interconnect;
    SystemServicesConfig system_services;

    // Validation
    bool validate() const;
    std::string get_validation_errors() const;

    // Utility functions
    size_t get_kpu_count() const;
    size_t get_gpu_count() const;
    size_t get_npu_count() const;
    const AcceleratorConfig* find_accelerator(const std::string& id) const;

    // Default configurations
    static SystemConfig create_minimal_kpu();
    static SystemConfig create_edge_ai();
    static SystemConfig create_datacenter();
};

} // namespace sw::sim
