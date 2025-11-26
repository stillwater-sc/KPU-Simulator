# Quick Start: domain_flow Integration

## TL;DR

```bash
# 1. Build domain_flow
cd ~/dev/domain_flow
cmake --preset user-ninja-release
cmake --build build/user-ninja-release

# 2. Build KPU-simulator
cd ~/dev/KPU-simulator
cmake -B build -DKPU_DOMAIN_FLOW_LOCAL_PATH=$HOME/dev/domain_flow -GNinja
cmake --build build

# 3. Copy test graphs
./scripts/copy_domain_flow_graphs.sh ~/dev/domain_flow

# 4. Run tests
cd build
ctest -R graph_loader -V

# 5. Use in code
# See examples below
```

## What You Can Do Now

### Load a .dfg Graph

```cpp
#include <sw/compiler/graph_loader.hpp>

auto graph = sw::kpu::compiler::load_graph("test_graphs/benchmarks/mobilenet_v1.dfg");
graph->print();
```

### Inspect Graph Structure

```cpp
std::cout << "Graph: " << graph->name << "\n";
std::cout << "Operators: " << graph->operators.size() << "\n";
std::cout << "Tensors: " << graph->tensors.size() << "\n";

for (const auto& op : graph->operators) {
    std::cout << "  " << op->name << " [" << static_cast<int>(op->type) << "]\n";
}
```

### Validate Graph

```cpp
if (graph->validate()) {
    std::cout << "Graph is valid\n";
} else {
    std::cerr << "Graph validation failed\n";
}
```

## File Extensions

- `.dfg` - domain_flow native format (use this)
- `.json` - JSON fallback (for testing without domain_flow)

## Build Options

### Option 1: Local Installation (Recommended for Development)
```bash
cmake -B build -DKPU_DOMAIN_FLOW_LOCAL_PATH=~/dev/domain_flow -GNinja
```

### Option 2: FetchContent (Automatic Download)
```bash
cmake -B build -GNinja  # Requires CMake 3.28+
```

### Option 3: No domain_flow (JSON Only)
```bash
cmake -B build -DKPU_USE_DOMAIN_FLOW=OFF -GNinja
```

## What's Implemented

✅ CMake integration with domain_flow
✅ Graph loading from .dfg files
✅ Operator conversion (domain_flow → KPU)
✅ JSON fallback format
✅ Unit tests
✅ Test graph infrastructure
✅ Helper scripts

## What's Next

The next step is **schedule generation** - converting graphs into DMA/BlockMover/Streamer command sequences.

This is the critical missing piece for the system-level compiler.

## Full Documentation

- **Complete Integration Guide**: `docs/domain-flow-integration.md`
- **Implementation Details**: `INTEGRATION_COMPLETE.md`
- **Test Graphs**: `test_graphs/README.md`
- **Native Format Details**: `NATIVE_FORMAT_INTEGRATION.md`

## Quick Test

```bash
# After building
cd build

# Test JSON loading (works without domain_flow)
./tests/compiler/graph_loader_test

# Test .dfg loading (requires domain_flow + test graphs)
# First copy graphs, then:
ctest -R graph_loader -V
```

## Success!

You now have a functional graph loader that can:
1. Load computational graphs from domain_flow
2. Convert operators to KPU representation
3. Validate graph structure
4. Serve as input to the schedule generator

Ready to move forward with system-level compiler development!
