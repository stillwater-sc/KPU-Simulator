# JSON Configuration System

## Overview

The Stillwater System Simulator now supports comprehensive JSON-based configuration for defining heterogeneous computing systems. This enables flexible specification of host processors, accelerators (KPU, GPU, NPU), memory subsystems, and interconnects.

## Key Features

### 1. Hierarchical Component Organization
- **Host + Host Memory**: CPU with DIMM-based DDR/LPDDR memory
- **Accelerators + Accelerator Memory**: Each accelerator (KPU/GPU/NPU) owns its memory subsystem (GDDR/HBM)
- **Interconnects**: PCIe, CXL, NoC, AMBA for component communication
- **System Services**: Memory management, interrupt handling, power management

### 2. Multiple Configuration Methods

**Factory Methods** - Predefined configurations:
```cpp
auto config = SystemConfig::create_minimal_kpu();
auto config = SystemConfig::create_edge_ai();
auto config = SystemConfig::create_datacenter();
```

**JSON Files** - External configuration:
```cpp
SystemSimulator sim("configs/examples/minimal_kpu.json");
sim.initialize();
```

**Programmatic** - Runtime configuration:
```cpp
SystemConfig config;
config.system.name = "My System";
// ... configure components ...
SystemSimulator sim(config);
```

### 3. Validation and Error Reporting
- Automatic validation on load
- Detailed error messages
- Sanity checks for component relationships

### 4. Memory Subsystem Modeling

Properly models the architectural difference between:
- **Host memory**: Commodity DIMMs (DDR4/DDR5/LPDDR)
- **Accelerator memory**: Custom layouts (GDDR6, HBM2/HBM3)
- **Form factors**: DIMM vs Substrate/Interposer/3DStack

## Architecture

### Component Hierarchy

```
SystemSimulator
├── SystemConfig
│   ├── SystemInfo (name, description)
│   ├── HostConfig
│   │   ├── CPUConfig
│   │   ├── HostMemoryConfig (DIMM-based)
│   │   │   ├── DRAMControllerConfig
│   │   │   └── MemoryModuleConfig[] (DDR/LPDDR)
│   │   └── StorageConfig[] (SSD/NVMe)
│   ├── AcceleratorConfig[]
│   │   ├── KPUConfig
│   │   │   ├── KPUMemoryConfig (GDDR/HBM on substrate)
│   │   │   ├── KPUComputeConfig (systolic arrays)
│   │   │   └── KPUDataMovementConfig (DMA/streamers)
│   │   ├── GPUConfig
│   │   │   └── GPUMemoryConfig (GDDR/HBM)
│   │   └── NPUConfig
│   │       └── NPUMemoryConfig (OnChip/LPDDR)
│   ├── InterconnectConfig
│   │   ├── HostToAcceleratorConfig (PCIe/CXL)
│   │   ├── AcceleratorToAcceleratorConfig (NoC/NVLink)
│   │   └── OnChipConfig (AMBA/CHI)
│   └── SystemServicesConfig
└── Component Instances
    └── KPUSimulator[] (instantiated from config)
```

## File Structure

### Configuration Files
- `configs/schema.md` - Complete JSON schema reference
- `configs/README.md` - User guide and best practices
- `configs/examples/`
  - `minimal_kpu.json` - Basic single-KPU system
  - `edge_ai.json` - KPU + NPU edge device
  - `datacenter_hbm.json` - High-end datacenter node

### Implementation
- `include/sw/system/system_config.hpp` - Configuration data structures
- `include/sw/system/config_loader.hpp` - JSON parsing/serialization
- `include/sw/system/toplevel.hpp` - System simulator with config support
- `src/system/system_config.cpp` - Configuration validation and factories
- `src/system/config_loader.cpp` - JSON I/O implementation
- `src/system/toplevel.cpp` - Component instantiation from config

### Tests and Examples
- `tests/system/test_system_config.cpp` - Comprehensive configuration tests
- `examples/basic/system_config_demo.cpp` - Interactive demo program

## Usage Examples

### Example 1: Load and Run from JSON

```cpp
#include <sw/system/toplevel.hpp>

int main() {
    sw::sim::SystemSimulator sim("configs/examples/minimal_kpu.json");

    if (!sim.initialize()) {
        return 1;
    }

    // Access KPU accelerator
    auto* kpu = sim.get_kpu(0);

    // Run workload...

    sim.shutdown();
    return 0;
}
```

### Example 2: Custom Configuration

```cpp
#include <sw/system/system_config.hpp>
#include <sw/system/toplevel.hpp>

int main() {
    using namespace sw::sim;

    SystemConfig config;
    config.system.name = "My Custom System";

    // Configure host
    config.host.cpu.core_count = 8;
    MemoryModuleConfig mem;
    mem.id = "ddr5_0";
    mem.type = "DDR5";
    mem.capacity_gb = 32;
    config.host.memory.modules.push_back(mem);

    // Add KPU
    AcceleratorConfig kpu_accel;
    kpu_accel.type = AcceleratorType::KPU;
    kpu_accel.id = "kpu_0";

    KPUConfig kpu;
    // ... configure KPU memory, compute, etc ...
    kpu_accel.kpu_config = kpu;
    config.accelerators.push_back(kpu_accel);

    // Validate and use
    if (!config.validate()) {
        std::cerr << config.get_validation_errors();
        return 1;
    }

    SystemSimulator sim(config);
    sim.initialize();
    // ...
}
```

### Example 3: Configuration Round-Trip

```cpp
#include <sw/system/config_loader.hpp>

// Create config
auto config = SystemConfig::create_edge_ai();

// Save to file
ConfigLoader::save_to_file(config, "my_config.json");

// Load back
auto loaded = ConfigLoader::load_from_file("my_config.json");

// Validate
if (!loaded.validate()) {
    auto errors = ConfigLoader::get_validation_errors("my_config.json");
    for (const auto& error : errors) {
        std::cout << error << "\n";
    }
}
```

## Design Rationale

### Why Separate Host and Accelerator Memory?

Modern heterogeneous systems have fundamentally different memory subsystems:

1. **Host Memory (DIMM-based)**
   - Standardized form factors (DIMM, SODIMM)
   - Commodity technology (DDR4/DDR5)
   - Optimized for capacity and cost
   - Pluggable modules

2. **Accelerator Memory (Custom layouts)**
   - Custom PCB/substrate layouts
   - High-bandwidth technologies (GDDR6, HBM)
   - Optimized for bandwidth and proximity to compute
   - Soldered or stacked directly on package

This architectural difference is reflected in the configuration schema, where each accelerator owns and configures its memory subsystem independently.

### JSON vs. Other Formats

**Why JSON?**
- Human-readable and editable
- Excellent tooling support (validators, editors)
- Native C++ support via nlohmann/json library
- Easy integration with Python and web tools

**Why not YAML/TOML?**
- JSON is sufficient for our hierarchical structure
- Simpler parsing with fewer edge cases
- Better IDE support in most environments

### Configuration Philosophy

1. **Explicit over Implicit**: All important parameters are specified
2. **Validation Early**: Catch errors before initialization
3. **Sensible Defaults**: Factory methods provide working configurations
4. **Flexibility**: Support both JSON files and programmatic configuration
5. **Extensibility**: Easy to add new accelerator types and parameters

## Performance Considerations

- Configuration loading is a one-time cost at initialization
- Parsed configurations are cached in C++ structures
- Component instantiation happens once based on configuration
- No runtime overhead once system is initialized

## Future Enhancements

Potential areas for expansion:

1. **More Accelerator Types**
   - DSP accelerators
   - FPGA configurations
   - Custom ASIC definitions

2. **Advanced Features**
   - Memory partitioning schemes
   - Power budgets and thermal models
   - Network topology definitions

3. **Tooling**
   - Configuration generator GUI
   - Validation tool with detailed reports
   - Configuration diff/merge utilities

4. **Python Integration**
   - Python bindings for configuration API
   - Jupyter notebook examples
   - Configuration generation from ML models

## References

- [Configuration README](../configs/README.md) - User guide
- [JSON Schema](../configs/schema.md) - Complete schema reference
- [Example Configurations](../configs/examples/) - Reference implementations
- [System Config Demo](../examples/basic/system_config_demo.cpp) - Interactive examples
- [Unit Tests](../tests/system/test_system_config.cpp) - Comprehensive test coverage

## Contributing

When adding new configuration options:

1. Update `system_config.hpp` with new structures
2. Add parsing/serialization in `config_loader.cpp`
3. Update validation logic in `system_config.cpp`
4. Add example to JSON schema and example configs
5. Write unit tests for new functionality
6. Update documentation

## Summary

The JSON configuration system provides a flexible, validated way to define complex heterogeneous computing systems. It properly models the architectural differences between host and accelerator memory subsystems, supports multiple configuration workflows, and includes comprehensive validation and error reporting.
