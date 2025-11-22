# Domain Flow Integration Guide

This document describes how to integrate the `domain_flow` IR from https://github.com/branes-ai/domain_flow into the KPU simulator for defining computational graphs and generating system-level schedules.

## Overview

The KPU simulator uses the `domain_flow` repository for:

1. **IR Definition** - Computational graph representation using MLIR
2. **Operator Semantics** - Systems of Uniform Recurrence Equations (SUREs)
3. **Graph Transformations** - Analysis and optimization of operator graphs

The integration allows:
- Loading graphs from MLIR files
- Converting framework graphs (PyTorch/TensorFlow/JAX) to KPU schedules
- Testing compiler behavior with well-defined test graphs

## Integration Methods

Three integration options are supported:

### Option 1: CMake FetchContent (Default)

Automatically downloads `domain_flow` during CMake configuration.

```cmake
# Automatically enabled by default
cmake --preset user-ninja-release
cmake --build build/user-ninja-release
```

The integration is controlled in `cmake/DomainFlowIntegration.cmake`:
- Downloads from GitHub at configure time
- Builds only necessary components
- Exposes includes to KPU simulator

### Option 2: Local Installation

Use a local build of `domain_flow` for development.

```bash
# Build domain_flow locally
cd ~/dev
git clone https://github.com/branes-ai/domain_flow.git
cd domain_flow
cmake --preset user-ninja-release
cmake --build build/user-ninja-release

# Configure KPU simulator to use local build
cd ~/dev/KPU-simulator
cmake -B build \
  -DKPU_DOMAIN_FLOW_LOCAL_PATH=/home/user/dev/domain_flow \
  -GNinja
cmake --build build
```

**When to use:**
- Developing domain_flow and KPU simulator simultaneously
- Testing domain_flow changes before committing
- Offline development

### Option 3: Disable domain_flow (JSON only)

Disable MLIR integration and use JSON test graphs only.

```bash
cmake -B build -DKPU_USE_DOMAIN_FLOW=OFF -GNinja
cmake --build build
```

**When to use:**
- CMake version < 3.28
- Quick builds without MLIR dependencies
- CI environments where domain_flow is unavailable

## Directory Structure

```
KPU-simulator/
├── cmake/
│   └── DomainFlowIntegration.cmake  # Integration logic
├── include/sw/compiler/
│   └── graph_loader.hpp             # Graph loading interface
├── src/compiler/
│   ├── CMakeLists.txt
│   └── graph_loader.cpp             # Implementation
├── test_graphs/                     # Test computational graphs
│   ├── README.md
│   ├── simple/                      # Basic operator tests
│   │   ├── matmul.mlir             # MLIR format (domain_flow)
│   │   ├── matmul.json             # JSON format (fallback)
│   │   ├── conv2d.mlir
│   │   └── elementwise.mlir
│   ├── networks/                    # Complete networks
│   │   ├── mlp.mlir
│   │   └── resnet_block.mlir
│   └── benchmarks/                  # Performance tests
│       ├── bert_encoder.mlir
│       └── transformer_layer.mlir
└── tests/compiler/
    ├── CMakeLists.txt
    └── test_graph_loader.cpp        # Unit tests
```

## Usage Examples

### C++ API

```cpp
#include <sw/compiler/graph_loader.hpp>

using namespace sw::kpu::compiler;

// Load graph from MLIR (domain_flow format)
auto graph = load_graph("test_graphs/simple/matmul.mlir");

// Inspect graph
graph->print();
std::cout << "Operators: " << graph->operators.size() << "\n";
std::cout << "Tensors: " << graph->tensors.size() << "\n";

// Validate
if (!graph->validate()) {
    std::cerr << "Graph validation failed\n";
    return;
}

// Get execution order
auto order = graph->get_execution_order();
for (auto* op : order) {
    std::cout << "Operator: " << op->name << "\n";
}

// TODO: Generate schedule from graph
// auto schedule = generate_schedule(graph, kpu_config);
```

### JSON Fallback (No domain_flow)

```cpp
// Load from JSON instead of MLIR
auto graph = load_graph("test_graphs/simple/matmul.json");

// Rest of API is identical
graph->print();
```

### Python API (Future)

```python
from stillwater_kpu.compiler import load_graph

# Load graph
graph = load_graph("test_graphs/simple/matmul.mlir")

# Generate schedule
from stillwater_kpu.compiler import generate_schedule
schedule = generate_schedule(graph, kpu_config)
```

## Test Graph Format

### MLIR Format (domain_flow)

```mlir
// test_graphs/simple/matmul.mlir
module @matmul_graph {
  func.func @matmul(%A: tensor<1024x512xf32>,
                    %B: tensor<512x1024xf32>) -> tensor<1024x1024xf32> {
    %C = "domain_flow.matmul"(%A, %B) {
      transpose_a = false,
      transpose_b = false
    } : (tensor<1024x512xf32>, tensor<512x1024xf32>) -> tensor<1024x1024xf32>
    return %C : tensor<1024x1024xf32>
  }
}
```

### JSON Format (Fallback)

```json
{
  "name": "simple_matmul",
  "tensors": {
    "A": {"shape": [1024, 512], "dtype": "float32"},
    "B": {"shape": [512, 1024], "dtype": "float32"},
    "C": {"shape": [1024, 1024], "dtype": "float32"}
  },
  "operators": [
    {
      "name": "matmul_0",
      "type": "MATMUL",
      "inputs": ["A", "B"],
      "outputs": ["C"]
    }
  ]
}
```

## Building and Testing

```bash
# Full build with domain_flow
cmake --preset user-ninja-release
cmake --build build/user-ninja-release

# Run compiler tests
ctest --test-dir build/user-ninja-release -R graph_loader

# Or run directly
./build/user-ninja-release/tests/compiler/graph_loader_test
```

## Adding New Test Graphs

1. Create MLIR file in `test_graphs/simple/` or `test_graphs/networks/`:
```mlir
module @my_graph {
  // Define your graph here
}
```

2. Add corresponding test case:
```cpp
TEST_CASE("Load my custom graph", "[compiler]") {
    auto graph = load_graph("test_graphs/simple/my_graph.mlir");
    REQUIRE(graph != nullptr);
    // Add assertions
}
```

3. Document in `test_graphs/README.md`

## Current Status and Roadmap

### ✅ Implemented

- [x] CMake integration infrastructure
- [x] Graph loader interface
- [x] JSON graph format (fallback)
- [x] Basic graph validation
- [x] Test infrastructure
- [x] Directory structure

### 🚧 In Progress

- [ ] Actual MLIR parsing using domain_flow APIs
- [ ] Operator type conversion (domain_flow → KPU)
- [ ] Tensor metadata extraction from MLIR

### 📋 Planned

- [ ] Schedule generation from graphs
- [ ] Integration with DMA/BlockMover/Streamer
- [ ] Bufferization and memory planning
- [ ] Python bindings for graph loading
- [ ] ONNX/PyTorch/JAX importers

## Troubleshooting

### CMake version error

```
Error: domain_flow requires CMake 3.28+
```

**Solution:** Either:
- Upgrade CMake: `pip install --upgrade cmake`
- Use local installation: `-DKPU_DOMAIN_FLOW_LOCAL_PATH=...`
- Disable domain_flow: `-DKPU_USE_DOMAIN_FLOW=OFF`

### MLIR parsing not implemented

```
DomainFlowGraphLoader::load() - STUB IMPLEMENTATION
```

**This is expected** - MLIR parsing integration is not yet complete. Use JSON format for testing:
```cpp
auto graph = load_graph("test_graphs/simple/matmul.json");
```

### Test graphs not found

```
Error: Failed to open file: test_graphs/simple/matmul.json
```

**Solution:** Tests expect to be run from project root:
```bash
cd /path/to/KPU-simulator
./build/tests/compiler/graph_loader_test
```

Or use CTest which sets working directory correctly:
```bash
ctest --test-dir build -R graph_loader
```

## Contributing

When adding domain_flow integration features:

1. Update `src/compiler/graph_loader.cpp` with MLIR parsing logic
2. Add test cases in `tests/compiler/test_graph_loader.cpp`
3. Create example graphs in `test_graphs/`
4. Document in this file

## References

- domain_flow repository: https://github.com/branes-ai/domain_flow
- KPU Architecture: `docs/kpu_architecture.md`
- MLIR Documentation: https://mlir.llvm.org/
- Compiler Architecture: `docs/compiler-architecture.md` (TODO)
