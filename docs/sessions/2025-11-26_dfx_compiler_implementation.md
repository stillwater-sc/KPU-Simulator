# Session Log: DFX Compiler Implementation

**Date:** 2025-11-26
**Focus:** Domain Flow Execution (DFX) layer and KPU kernel compiler implementation

---

## Session Goals

1. Create system-level schedules for key operators (matmul → conv2d → mlp with softmax)
2. Build a `kpu_kernel_compiler` that takes a DFG and produces an object file
3. Create a loader program to program the KPU simulator
4. Define PTX-equivalent abstraction layer for hardware-agnostic representation
5. Reorganize tools directory from `tools/cpp` to proper categories

---

## Major Accomplishments

### 1. Domain Flow Execution (DFX) Layer

Created a PTX-equivalent hardware-agnostic intermediate representation for KPU programs.

**Key Design Principles:**
- **Hardware-agnostic**: Same DFX works on different KPU configurations
- **Expressive**: Captures all necessary scheduling decisions
- **Optimizable**: Allows driver-level optimization
- **Serializable**: Can be saved to disk and loaded later

**Files Created:**
- `include/sw/compiler/dfx/dfx.hpp` - Core DFX types
- `include/sw/compiler/dfx/dfx_object_file.hpp` - JSON serialization

**Key Types:**
```cpp
namespace sw::kpu::compiler::dfx {
    // Data types
    enum class DataType { FLOAT32, FLOAT16, BFLOAT16, INT32, INT16, INT8, UINT8, BOOL };

    // Memory hierarchy levels
    enum class MemoryLevel { EXTERNAL, L3, L2, L1, REGISTER };

    // Dataflow strategies
    enum class DataflowStrategy { OUTPUT_STATIONARY, WEIGHT_STATIONARY, INPUT_STATIONARY };

    // Operations
    struct DataMoveOp : Operation { /* DRAM↔L3↔L2↔L1 transfers */ };
    struct ComputeOp : Operation { /* MATMUL, CONV2D, etc. */ };
    struct BarrierOp : Operation { /* Synchronization */ };

    // Complete program
    struct Program {
        std::vector<TensorDescriptor> tensors;
        std::vector<std::unique_ptr<Operation>> operations;
        TilingConfig tiling;
        DataflowStrategy dataflow;
        PerformanceHints hints;
    };
}
```

### 2. KPU Kernel Compiler

Built a complete compilation pipeline from Domain Flow Graph (DFG) to KPU object files.

**Tool:** `kpu-kernel-compiler`

**Usage:**
```bash
kpu-kernel-compiler matmul.dfg -o matmul.kpu
kpu-kernel-compiler matmul.json --emit-dfx --verbose
kpu-kernel-compiler conv2d.dfg -d weight-stationary -o conv2d.kpu
```

**Files Created:**
- `tools/compiler/kpu-kernel-compiler/main.cpp` - CLI entry point
- `tools/compiler/kpu-kernel-compiler/dfg_parser.hpp/cpp` - DFG/JSON parsing
- `tools/compiler/kpu-kernel-compiler/dfx_generator.hpp/cpp` - DFX generation
- `tools/compiler/kpu-kernel-compiler/object_writer.hpp/cpp` - .kpu file writer

**Test Results:**
```
$ ./kpu-kernel-compiler test_graphs/simple/matmul.json -v
KPU Kernel Compiler v1.0.0
Generating DFX...
  Matrix: 1024x1024x512
  Tiles: 80x80x48
  Tile grid: 13x13x11
  Operations: 13689
  Data moves: 11830
  Computes: 1859
```

### 3. KPU Loader Framework (Skeleton)

Created the framework for loading and binding DFX programs to hardware resources.

**Files Created:**
- `tools/runtime/kpu-loader/main.cpp` - CLI entry point (stub)
- `tools/runtime/kpu-loader/object_reader.hpp/cpp` - .kpu file reader
- `tools/runtime/kpu-loader/schedule_binder.hpp/cpp` - Resource binding

**Key Structures:**
```cpp
struct BoundOperation {
    const dfx::Operation* dfx_op;
    size_t dma_engine_id;
    size_t block_mover_id;
    size_t streamer_id;
    uint64_t source_addr;
    uint64_t dest_addr;
    uint64_t start_cycle;
    uint64_t end_cycle;
};

struct BoundSchedule {
    const dfx::Program* program;
    std::vector<BoundOperation> operations;
    uint64_t total_cycles;
    double estimated_throughput;  // TFLOPS
};
```

### 4. Tools Directory Reorganization

Reorganized from flat `tools/cpp` to category-based structure:

```
tools/
├── compiler/           # kpu-kernel-compiler
├── runtime/            # kpu-loader, kpu-driver
├── analysis/           # kpu-profiler, kpu-trace-analyzer
├── development/        # kpu-assembler, kpu-debugger, kpu-disassembler
├── configuration/      # kpu-config
├── benchmark/          # kpu-benchmark
└── dse/                # Design space exploration
```

Added `kpu_add_tool()` CMake helper function for consistent tool creation.

### 5. Naming Refactor: KIR → DFX

Renamed the intermediate representation from "KIR" (KPU Intermediate Representation) to "DFX" (Domain Flow Execution) for marketing as a PTX replacement.

**Changes:**
- Namespace: `kir` → `dfx`
- Directory: `include/sw/compiler/kir/` → `include/sw/compiler/dfx/`
- Files: `kir.hpp` → `dfx.hpp`, `object_file.hpp` → `dfx_object_file.hpp`
- Class: `KIRGenerator` → `DFXGenerator`
- CLI: `--emit-kir` → `--emit-dfx`
- JSON: `"kir_version"` → `"dfx_version"`

---

## Object File Format (.kpu)

The .kpu file is a JSON-based format containing:

```json
{
  "dfx_version": "1.0.0",
  "name": "simple_matmul",
  "dataflow": "output_stationary",
  "tiling": {
    "Ti": 80, "Tj": 80, "Tk": 48,
    "num_tiles_m": 13, "num_tiles_n": 13, "num_tiles_k": 11
  },
  "tensors": [
    {"name": "A", "shape": [1024, 512], "dtype": "f32"},
    {"name": "B", "shape": [512, 1024], "dtype": "f32"},
    {"name": "C", "shape": [1024, 1024], "dtype": "f32", "is_output": true}
  ],
  "operations": [
    {"op_id": 1, "type": "DATA_MOVE", "move_type": "LOAD", ...},
    {"op_id": 7, "type": "COMPUTE", "compute_type": "MATMUL_TILE", ...}
  ],
  "hints": {
    "estimated_dram_bytes": 4664320,
    "arithmetic_intensity": 230.2,
    "estimated_compute_cycles": 2097152
  }
}
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KPU Compilation Pipeline                          │
└─────────────────────────────────────────────────────────────────────┘

  .dfg/.json ──▶ kpu-kernel-compiler ──▶ .kpu ──▶ kpu-loader ──▶ KPU Simulator
                        │                              │
                        ▼                              ▼
              ┌─────────────────┐          ┌─────────────────────┐
              │ Hardware-       │          │ Micro-Architecture  │
              │ Agnostic DFX    │          │ Specific Binding    │
              │ Representation  │          │ & Scheduling        │
              └─────────────────┘          └─────────────────────┘
```

**Layer 1: DFG** - High-level operator graph (MATMUL, CONV2D, etc.)
**Layer 2: DFX** - Tiled computation, abstract data movement, dataflow strategy
**Layer 3: .kpu** - Serialized DFX program
**Layer 4: Driver** - Concrete addresses, DMA/BlockMover/Streamer binding, cycle timing

---

## Files Created/Modified

### New Files
- `include/sw/compiler/dfx/dfx.hpp`
- `include/sw/compiler/dfx/dfx_object_file.hpp`
- `tools/compiler/kpu-kernel-compiler/main.cpp`
- `tools/compiler/kpu-kernel-compiler/dfg_parser.hpp`
- `tools/compiler/kpu-kernel-compiler/dfg_parser.cpp`
- `tools/compiler/kpu-kernel-compiler/dfx_generator.hpp`
- `tools/compiler/kpu-kernel-compiler/dfx_generator.cpp`
- `tools/compiler/kpu-kernel-compiler/object_writer.hpp`
- `tools/compiler/kpu-kernel-compiler/object_writer.cpp`
- `tools/compiler/CMakeLists.txt`
- `tools/runtime/kpu-loader/main.cpp`
- `tools/runtime/kpu-loader/object_reader.hpp`
- `tools/runtime/kpu-loader/object_reader.cpp`
- `tools/runtime/kpu-loader/schedule_binder.hpp`
- `tools/runtime/kpu-loader/schedule_binder.cpp`
- `docs/compiler/KPU_COMPILER_IMPLEMENTATION_PLAN.md`

### Modified Files
- `CMakeLists.txt` - Added `add_subdirectory(tools)` with `KPU_BUILD_TOOLS` option
- `tools/CMakeLists.txt` - Added `kpu_add_tool()` helper and tool categories
- `tools/python/CMakeLists.txt` - Created empty CMakeLists for Python tools

---

## Build and Test

```bash
# Configure
cmake --preset release

# Build kernel compiler
ninja kernel-compiler

# Test help
./tools/compiler/kpu-kernel-compiler --help

# Compile a matmul
./tools/compiler/kpu-kernel-compiler test_graphs/simple/matmul.json -o matmul.kpu -v
```

---

## Next Steps

1. **Complete the loader** - Implement full schedule binding and simulator execution
2. **Add CONV2D support** - Im2col transformation and strided data movement
3. **Multi-operator graphs** - Handle multiple operators with intermediate tensors
4. **MLP with Softmax** - Operator fusion and softmax tiling
5. **Binary format** - Add FlatBuffers/Cap'n Proto for production use
6. **Prefetching** - Implement prefetch operations in DFX

---

## References

- Implementation Plan: `docs/compiler/KPU_COMPILER_IMPLEMENTATION_PLAN.md`
- Example DFG: `test_graphs/simple/matmul.json`
- TileOptimizer: `src/compiler/tile_optimizer.cpp`
- L2TileScheduler: `src/compiler/l2_tile_scheduler.cpp`
