# System Configuration Guide

The Stillwater System Simulator uses JSON configuration files to define heterogeneous computing systems. This guide explains how to create and use configuration files.

## Overview

Configuration files define:
- **Host** processor and memory subsystem
- **Accelerators** (KPU, GPU, NPU) with their dedicated memory
- **Interconnects** between components
- **System services** (memory management, interrupts, power management)

## Quick Start

### Using Predefined Configurations

```cpp
#include <sw/system/toplevel.hpp>

// Use factory methods for common configurations
auto config = sw::sim::SystemConfig::create_minimal_kpu();
sw::sim::SystemSimulator sim(config);
sim.initialize();
sim.run_self_test();
```

### Loading from JSON File

```cpp
#include <sw/system/toplevel.hpp>

sw::sim::SystemSimulator sim("configs/examples/minimal_kpu.json");
sim.initialize();

// Access KPU accelerators
auto* kpu = sim.get_kpu(0);
auto* kpu_by_id = sim.get_kpu_by_id("kpu_0");
```

### Programmatic Configuration

```cpp
#include <sw/system/system_config.hpp>
#include <sw/system/config_loader.hpp>

using namespace sw::sim;

SystemConfig config;
config.system.name = "My Custom System";

// Configure host
config.host.cpu.core_count = 8;
MemoryModuleConfig host_mem;
host_mem.id = "ddr5_0";
host_mem.type = "DDR5";
host_mem.capacity_gb = 32;
host_mem.bandwidth_gbps = 51.2f;
config.host.memory.modules.push_back(host_mem);

// Add KPU accelerator
AcceleratorConfig kpu_accel;
kpu_accel.type = AcceleratorType::KPU;
kpu_accel.id = "kpu_0";

KPUConfig kpu;
// Configure KPU memory, compute, data movement...
kpu_accel.kpu_config = kpu;
config.accelerators.push_back(kpu_accel);

// Validate and use
if (config.validate()) {
    SystemSimulator sim(config);
    sim.initialize();
}
```

## Configuration File Structure

### Minimal Example

```json
{
  "system": {
    "name": "My System"
  },
  "host": {
    "cpu": {
      "core_count": 4,
      "frequency_mhz": 2400
    },
    "memory": {
      "modules": [
        {
          "id": "mem_0",
          "type": "DDR4",
          "capacity_gb": 16,
          "bandwidth_gbps": 25.6
        }
      ]
    }
  },
  "accelerators": [
    {
      "type": "KPU",
      "id": "kpu_0",
      "kpu_config": {
        "memory": {
          "type": "GDDR6",
          "banks": [
            { "id": "bank_0", "capacity_mb": 1024, "bandwidth_gbps": 100 }
          ],
          "scratchpads": [
            { "id": "scratch_0", "capacity_kb": 64 }
          ]
        },
        "compute_fabric": {
          "tiles": [
            {
              "id": "tile_0",
              "type": "systolic",
              "systolic_rows": 16,
              "systolic_cols": 16
            }
          ]
        },
        "data_movement": {
          "dma_engines": [
            { "id": "dma_0", "bandwidth_gbps": 50 }
          ]
        }
      }
    }
  ]
}
```

## Configuration Components

### Host Configuration

Defines the host processor and its DIMM-based memory subsystem:

```json
"host": {
  "cpu": {
    "core_count": 64,
    "frequency_mhz": 3500,
    "cache_l1_kb": 64,
    "cache_l2_kb": 1024,
    "cache_l3_kb": 262144
  },
  "memory": {
    "dram_controller": {
      "channel_count": 8,
      "data_width_bits": 64
    },
    "modules": [
      {
        "id": "dimm_0",
        "type": "DDR5",
        "form_factor": "DIMM",
        "capacity_gb": 128,
        "frequency_mhz": 5600,
        "bandwidth_gbps": 89.6
      }
    ]
  },
  "storage": [
    {
      "id": "nvme_0",
      "type": "NVME",
      "capacity_gb": 2048,
      "read_bandwidth_mbps": 7000,
      "write_bandwidth_mbps": 5000
    }
  ]
}
```

### KPU Accelerator Configuration

Each KPU has its own memory subsystem (typically GDDR or HBM):

```json
{
  "type": "KPU",
  "id": "kpu_0",
  "description": "High-bandwidth KPU with HBM",
  "kpu_config": {
    "memory": {
      "type": "HBM3",
      "form_factor": "Interposer",
      "banks": [
        {
          "id": "bank_0",
          "capacity_mb": 8192,
          "bandwidth_gbps": 819,
          "latency_ns": 10
        }
      ],
      "l3_tiles": [...],
      "l2_banks": [...],
      "scratchpads": [...]
    },
    "compute_fabric": {
      "tiles": [
        {
          "id": "tile_0",
          "type": "systolic",
          "systolic_rows": 32,
          "systolic_cols": 32,
          "datatype": "fp32"
        }
      ]
    },
    "data_movement": {
      "dma_engines": [...],
      "block_movers": [...],
      "streamers": [...]
    }
  }
}
```

### GPU Configuration

```json
{
  "type": "GPU",
  "id": "gpu_0",
  "gpu_config": {
    "compute_units": 128,
    "clock_mhz": 2100,
    "memory": {
      "type": "HBM3",
      "form_factor": "Interposer",
      "capacity_gb": 80,
      "bandwidth_gbps": 3200
    }
  }
}
```

### NPU Configuration

```json
{
  "type": "NPU",
  "id": "npu_0",
  "npu_config": {
    "tops_int8": 40,
    "tops_fp16": 20,
    "memory": {
      "type": "OnChip",
      "capacity_mb": 16,
      "bandwidth_gbps": 200
    }
  }
}
```

### Interconnect Configuration

```json
"interconnect": {
  "host_to_accelerator": {
    "type": "PCIe",
    "pcie_config": {
      "generation": 5,
      "lanes": 16,
      "bandwidth_gbps": 128
    }
  },
  "accelerator_to_accelerator": {
    "type": "NoC",
    "noc_config": {
      "topology": "mesh",
      "router_count": 4,
      "link_bandwidth_gbps": 100
    }
  },
  "on_chip": {
    "type": "AMBA",
    "amba_config": {
      "protocol": "CHI"
    }
  }
}
```

## Memory Technology Guidelines

### Host Memory (DIMM-based)
- **DDR4/DDR5**: Standard server/desktop memory
- **LPDDR4/LPDDR5**: Mobile and embedded systems
- **Form factors**: DIMM, SODIMM, LPDIMM, OnPackage

### Accelerator Memory (Custom layouts)
- **GDDR6/GDDR6X**: High-bandwidth graphics memory
- **HBM2/HBM3**: Ultra-high bandwidth stacked memory
- **LPDDR5**: Power-efficient accelerator memory
- **Form factors**: PCB, Substrate, Interposer, 3DStack

## Validation

Configuration files are automatically validated when loaded:

```cpp
#include <sw/system/config_loader.hpp>

// Check if file is valid
if (sw::sim::ConfigLoader::validate_file("my_config.json")) {
    // Load it
}

// Get detailed validation errors
auto errors = sw::sim::ConfigLoader::get_validation_errors("my_config.json");
for (const auto& error : errors) {
    std::cout << error << "\n";
}
```

Validation checks:
- Required fields are present
- Component counts are reasonable
- Memory capacities and bandwidths are non-zero
- Interconnects match system topology
- Accelerators have required type-specific configuration

## Example Use Cases

### 1. Edge AI Device
- Low-power host with LPDDR5
- Small KPU with LPDDR memory
- NPU for CNN inference
- See: `examples/edge_ai.json`

### 2. Datacenter Node
- High-performance host with DDR5
- KPU with HBM3 for large models
- GPU for training workloads
- PCIe Gen5 interconnect
- See: `examples/datacenter_hbm.json`

### 3. Development System
- Minimal host configuration
- Single KPU for testing
- Basic interconnect
- See: `examples/minimal_kpu.json`

## Best Practices

1. **Start with examples**: Copy and modify existing configurations
2. **Validate early**: Check configuration before full initialization
3. **Match bandwidths**: Ensure interconnect doesn't bottleneck memory
4. **Use appropriate memory**: Match memory technology to use case
5. **Document custom configs**: Add description fields for clarity

## API Reference

See:
- `include/sw/system/system_config.hpp` - Configuration data structures
- `include/sw/system/config_loader.hpp` - JSON loading/saving
- `include/sw/system/toplevel.hpp` - System simulator interface
- `configs/schema.md` - Complete JSON schema reference

## Troubleshooting

**Problem**: Configuration fails to load
- Check JSON syntax (use a JSON validator)
- Verify all required fields are present
- Check file path is correct

**Problem**: Validation fails
- Run `get_validation_errors()` to see specific issues
- Check that all counts are > 0
- Ensure memory modules are defined

**Problem**: System doesn't initialize
- Check console output for specific component errors
- Verify accelerator configurations are complete
- Ensure sufficient memory for requested configuration
