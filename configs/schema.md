# System Configuration JSON Schema

## Overview
The system simulator uses JSON configuration files to define heterogeneous computing systems composed of processing elements with their tightly-coupled memory subsystems, connected via system interconnects.

## Architecture Principles

1. **Component Grouping**: Processing elements (host CPU, accelerators) own their memory subsystems
2. **Memory Coupling**: Each processor type has its own memory technology optimized for its workload
3. **Interconnect**: System-level fabrics connect the major component groups

## Schema Structure

```json
{
  "system": {
    "name": "string",              // Human-readable system name
    "description": "string",       // Optional description
    "clock_frequency_mhz": integer // System clock (optional)
  },

  "host": {
    "cpu": {
      "core_count": integer,
      "frequency_mhz": integer,
      "cache_l1_kb": integer,
      "cache_l2_kb": integer,
      "cache_l3_kb": integer
    },
    "memory": {
      "dram_controller": {
        "channel_count": integer,
        "data_width_bits": integer
      },
      "modules": [
        {
          "id": "string",
          "type": "DDR4|DDR5|LPDDR4|LPDDR5",
          "form_factor": "DIMM|SODIMM|LPDIMM|OnPackage",
          "capacity_gb": integer,
          "frequency_mhz": integer,
          "bandwidth_gbps": integer,
          "latency_ns": integer,
          "channels": integer
        }
      ]
    },
    "storage": [
      {
        "id": "string",
        "type": "SSD|NVME|HDD",
        "capacity_gb": integer,
        "read_bandwidth_mbps": integer,
        "write_bandwidth_mbps": integer,
        "latency_us": integer
      }
    ]
  },

  "accelerators": [
    {
      "type": "KPU|GPU|NPU|DSP|FPGA",
      "id": "string",
      "description": "string",

      // KPU-specific configuration
      "kpu_config": {
        "memory": {
          "type": "GDDR6|HBM2|HBM3|Custom",
          "form_factor": "Substrate|PCB|Interposer|3DStack",
          "banks": [
            {
              "id": "string",
              "capacity_mb": integer,
              "bandwidth_gbps": integer,
              "latency_ns": integer
            }
          ],
          "l3_tiles": [
            {
              "id": "string",
              "capacity_kb": integer
            }
          ],
          "l2_banks": [
            {
              "id": "string",
              "capacity_kb": integer
            }
          ],
          "scratchpads": [
            {
              "id": "string",
              "capacity_kb": integer
            }
          ]
        },
        "compute_fabric": {
          "tiles": [
            {
              "id": "string",
              "type": "systolic|vector|scalar",
              "systolic_rows": integer,
              "systolic_cols": integer,
              "datatype": "fp32|fp16|int8|bfloat16"
            }
          ]
        },
        "data_movement": {
          "dma_engines": [
            {
              "id": "string",
              "bandwidth_gbps": integer,
              "channels": integer
            }
          ],
          "block_movers": [
            {
              "id": "string"
            }
          ],
          "streamers": [
            {
              "id": "string"
            }
          ]
        }
      },

      // GPU-specific configuration
      "gpu_config": {
        "compute_units": integer,
        "clock_mhz": integer,
        "memory": {
          "type": "GDDR6|GDDR6X|HBM2|HBM2E|HBM3",
          "form_factor": "PCB|Substrate|Interposer",
          "capacity_gb": integer,
          "bandwidth_gbps": integer,
          "bus_width_bits": integer
        }
      },

      // NPU-specific configuration
      "npu_config": {
        "tops_int8": integer,
        "tops_fp16": integer,
        "memory": {
          "type": "LPDDR5|HBM|OnChip",
          "capacity_mb": integer,
          "bandwidth_gbps": integer
        }
      }
    }
  ],

  "interconnect": {
    "host_to_accelerator": {
      "type": "PCIe|CXL|NVLink|CustomFabric",
      "pcie_config": {
        "generation": integer,
        "lanes": integer,
        "bandwidth_gbps": integer
      },
      "cxl_config": {
        "version": "1.0|2.0|3.0",
        "bandwidth_gbps": integer
      }
    },
    "accelerator_to_accelerator": {
      "type": "NVLink|InfinityFabric|NoC|None",
      "noc_config": {
        "topology": "mesh|torus|ring|crossbar",
        "router_count": integer,
        "link_bandwidth_gbps": integer
      }
    },
    "on_chip": {
      "type": "AMBA|CHI|TileLink|Custom",
      "amba_config": {
        "protocol": "AXI4|AXI5|ACE|CHI"
      }
    },
    "network": {
      "enabled": boolean,
      "type": "Ethernet|RoCE|InfiniBand",
      "speed_gbps": integer
    }
  },

  "system_services": {
    "memory_manager": {
      "enabled": boolean,
      "pool_size_mb": integer,
      "alignment_bytes": integer
    },
    "interrupt_controller": {
      "enabled": boolean
    },
    "power_management": {
      "enabled": boolean
    }
  }
}
```

## Key Design Decisions

### 1. Memory Subsystem Ownership
- **Host**: DDR/LPDDR modules on DIMMs, optimized for capacity and cost
- **KPU**: GDDR/HBM on substrate/interposer, optimized for bandwidth
- **GPU**: Similar to KPU, high-bandwidth memory close to compute
- **NPU**: Often LPDDR for power efficiency or on-chip SRAM

### 2. Form Factors Matter
- **DIMM/SODIMM**: Standardized modules for host memory
- **Substrate**: Custom memory layout for accelerators
- **Interposer**: 2.5D integration (HBM)
- **3DStack**: 3D stacked memory (HBM3)

### 3. Accelerator Array
Multiple accelerators of different types can coexist, each with dedicated memory

### 4. Hierarchical Interconnect
- **Host ↔ Accelerator**: PCIe, CXL (for memory coherence)
- **Accelerator ↔ Accelerator**: High-speed links or NoC
- **On-chip**: AMBA/CHI for internal communication

## Validation Rules

1. **Component Ownership**: Each accelerator must define its memory subsystem
2. **Interconnect Consistency**: If multiple accelerators exist, accelerator-to-accelerator interconnect should be defined
3. **Bandwidth Matching**: Interconnect bandwidth should not create obvious bottlenecks
4. **Memory Technology Constraints**:
   - DDR/LPDDR: Host only
   - GDDR: Accelerators only
   - HBM: Accelerators only (GPUs, high-end KPUs)
   - OnChip: NPUs and embedded processors

## Example Configurations

See `configs/examples/` for:
- `minimal_kpu.json` - Single KPU with basic memory
- `edge_ai.json` - KPU + NPU for edge devices
- `datacenter.json` - Host + multiple GPUs + KPU
- `hpc.json` - High-performance computing with HBM
