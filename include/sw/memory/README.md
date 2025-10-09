# Memory Subsystem

This directory defines the interfaces of the external memory subsystem of a compute engine that utilizes DDR, GDDR, LPDDR, or HBM external memory modules.

## Overview

A typical system simulator is constructed by creating the constituent components and wiring them up according to the system under test.

For example, if we want to model a single Chip KPU solution, with its own local HBM memory, and connected to a host processor via PCIe,
we would create a simulator with the following components:

1. CPU to model the host processor
2. External Memory connected to the host
3. KPU to model the accelerator
4. Local Memory connected to the KPU
5. PCIe bridge to connect CPU and KPU subsystems to each other

## Components

### Core Memory Modules

- **memory_map.hpp**: Cross-platform memory mapping abstraction (Windows/Linux/macOS)
  - Sparse virtual memory allocation
  - On-demand page commitment via OS kernel
  - Statistics and performance monitoring

- **sparse_memory.hpp**: Sparse memory manager with page tracking
  - Manages very large virtual address spaces (up to 48-bit addressing)
  - Tracks accessed pages for statistics
  - Thread-safe operations

- **external_memory.hpp**: Main external memory interface
  - Dual backend support (dense/sparse)
  - Automatic backend selection based on size
  - Compatible with all memory types (DDR/GDDR/LPDDR/HBM)

- **cache.hpp**: Cache hierarchy components
  - L2 and L3 cache modeling
  - Configurable cache policies

## Sparse Memory Support

The memory subsystem now supports **sparse memory allocation** for simulating very large configurations:

### Usage Example

```cpp
#include <sw/memory/external_memory.hpp>

// Datacenter configuration - 256GB HBM3
ExternalMemory::Config config;
config.capacity_mb = 256 * 1024;   // 256GB virtual space
config.bandwidth_gbps = 3200;      // 3.2TB/s bandwidth
config.auto_backend = true;        // Auto-select sparse backend

ExternalMemory mem(config);
// Physical memory: only what you actually use!
// Virtual space: full 256GB addressable
```

### Key Features

1. **On-Demand Allocation**: Memory pages allocated only when written to
2. **Cross-Platform**: Works on Windows, Linux, and macOS
3. **Large Address Spaces**: Support for 48-bit addressing (256TB)
4. **Transparent**: Same API as dense memory
5. **Statistics**: Track actual physical memory usage

### Backend Selection

- **Dense Backend** (< 1GB): Uses `std::vector`, all memory upfront
- **Sparse Backend** (≥ 1GB): Uses memory mapping, on-demand allocation
- **Auto-selection**: Automatically chooses best backend based on size

## Documentation

For detailed information about the sparse memory implementation, see:
- [Sparse Memory Implementation](../../docs/sparse-memory-implementation.md)

## Testing

Memory subsystem tests are located in `tests/memory/`:
- `test_memory_map.cpp` - Memory mapping tests
- `test_sparse_memory.cpp` - Sparse memory manager tests
- `test_external_memory_sparse.cpp` - External memory integration tests

Run tests with:
```bash
cd build
ctest -L memory
```