# Test Graphs for KPU Simulator

This directory contains computational graph definitions using the domain_flow IR from https://github.com/branes-ai/domain_flow.

## Graph Formats

### domain_flow Native Format (Recommended)
The primary format is domain_flow's lightweight serialization format (.dfg files). This format:
- Does NOT require LLVM/MLIR dependencies
- Is compact and fast to parse
- Contains complete operator and tensor information
- Supports SUREs (Systems of Uniform Recurrence Equations)

### JSON Format (Fallback/Testing)
A simple JSON format (.json files) is also supported for:
- Quick testing without domain_flow
- Simple examples and documentation
- When domain_flow is not available

## Directory Structure

```
test_graphs/
├── README.md                    # This file
├── simple/                      # Simple operator graphs for basic testing
│   ├── matmul.dfg              # Simple matrix multiplication (domain_flow)
│   ├── matmul.json             # Simple matrix multiplication (JSON)
│   ├── conv2d.dfg              # Simple 2D convolution
│   └── elementwise.dfg         # Element-wise operations
├── networks/                    # Complete neural network graphs
│   ├── mlp.dfg                 # Multi-layer perceptron
│   ├── resnet_block.dfg        # ResNet residual block
│   └── transformer_layer.dfg   # Transformer layer
└── benchmarks/                  # Benchmark graphs for performance testing
    ├── bert_encoder.dfg        # BERT encoder layer
    ├── mobilenet_v2.dfg        # MobileNet V2 block
    └── efficientnet.dfg        # EfficientNet block
```

## Usage

### Loading a Graph in C++

```cpp
#include <sw/compiler/graph_loader.hpp>

// Load graph from domain_flow native format
auto graph = sw::kpu::compiler::load_graph("test_graphs/simple/matmul.dfg");

// Or use JSON for testing
auto graph_json = sw::kpu::compiler::load_graph("test_graphs/simple/matmul.json");

// Generate schedule
auto schedule = sw::kpu::compiler::generate_schedule(graph, kpu_config);

// Execute on simulator
simulator.execute_schedule(schedule);
```

### Loading a Graph in Python

```python
from stillwater_kpu.compiler import load_graph, generate_schedule

# Load graph from domain_flow format
graph = load_graph("test_graphs/simple/matmul.dfg")

# Or use JSON
graph_json = load_graph("test_graphs/simple/matmul.json")

# Generate and execute schedule
schedule = generate_schedule(graph, kpu_config)
simulator.execute_schedule(schedule)
```

## Graph Format

Graphs use domain_flow's native serialization format. Each graph contains:

1. **Operators**: Computational nodes (GEMM, Conv, etc.) as SUREs
2. **Tensors**: Data dependencies between operators
3. **Metadata**: Shape, dtype, layout information
4. **Attributes**: Operator-specific parameters

## Adding New Graphs

1. Create graph in domain_flow and save to `.dfg` file:
```cpp
// In domain_flow code (see domain_flow/workloads for examples)
using namespace sw::dfa;
DomainFlowGraph graph("my_graph");

// Build graph...
// graph.addNode(...);

// Save to file
graph.save("my_graph.dfg");
```

Or copy from domain_flow's test graphs:
```bash
cp ~/dev/domain_flow/data/dfg/mobilenet_v1.dfg test_graphs/benchmarks/
```

2. Or create JSON file manually for simple cases:
```json
{
  "name": "my_graph",
  "tensors": { /* tensor definitions */ },
  "operators": [ /* operator definitions */ ]
}
```

3. Add corresponding test case in `tests/compiler/test_graph_loader.cpp`
4. Document expected behavior and performance characteristics

## Integration with domain_flow

These graphs use domain_flow's native serialization which represents operators as Systems of Uniform Recurrence Equations (SUREs). The KPU compiler:

1. **Ingests** the domain_flow format (no LLVM/MLIR dependency)
2. **Analyzes** data dependencies
3. **Schedules** data movement (DMA/BlockMover/Streamer)
4. **Generates** executable schedules for the KPU simulator

Benefits of native format:
- **Lightweight**: No LLVM/MLIR linking required
- **Fast**: Compact binary format
- **Complete**: Full SURE representation
- **Portable**: Works in standalone environments

See [docs/domain-flow-integration.md](../docs/domain-flow-integration.md) for details.
