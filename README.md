# Stillwater Knowledge Processing Unit (KPU) Simulator

A high-performance C++20 functional simulator for the Stillwater KPU - a specialized hardware accelerator for knowledge processing and AI workloads. Features Python bindings for easy testing and education.

## Quick Start

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt install build-essential cmake ninja-build libomp-dev python3-dev python3-pip

# macOS
brew install cmake ninja libomp python3
xcode-select --install

# Windows
# Install Visual Studio 2022 with C++ support
# Install CMake and Python from official websites
```

### Build with CMake Presets (Recommended)

The project uses **CMake presets** for streamlined configuration. Choose the preset for your platform:

```bash
# Linux (default: GCC with Ninja)
cmake --preset=release
cmake --build --preset=release

# Linux with Clang
cmake --preset=linux-clang
cmake --build build

# Windows (Visual Studio)
cmake --preset=windows-msvc
cmake --build build --config Release

# macOS (Xcode)
cmake --preset=macos
cmake --build build --config Release

# Debug build (with sanitizers)
cmake --preset=debug
cmake --build --preset=debug
```

**Note:** Most presets use the **Ninja** build system. If you don't have Ninja installed:
- **Ubuntu/Debian:** `sudo apt install ninja-build`
- **macOS:** `brew install ninja`
- **Windows:** Download from [Ninja releases](https://github.com/ninja-build/ninja/releases) or use Visual Studio preset

**Alternative without Ninja:**
```bash
# Use Unix Makefiles or Visual Studio instead
cmake -B build -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
cmake --build build
```

#### Available Configure Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `release` | Optimized release build | Production use |
| `debug` | Debug build with sanitizers | Development & debugging |
| `minimal` | Core components only | Minimal installation |
| `full` | All features enabled (CUDA, OpenCL, docs) | Full-featured build |
| `linux-gcc` | Linux with GCC | Linux development |
| `linux-clang` | Linux with Clang | Linux with Clang toolchain |
| `windows-msvc` | Windows with MSVC | Windows development |
| `macos` | macOS with Xcode | macOS development |

### Advanced Build Options

CMake presets automatically configure sensible defaults. To customize:

```bash
# With domain_flow integration (local installation)
cmake --preset=release -DKPU_DOMAIN_FLOW_LOCAL_PATH=~/dev/domain_flow
cmake --build --preset=release

# Minimal build (no tests, examples, or Python)
cmake --preset=minimal
cmake --build build

# Full build with all features
cmake --preset=full
cmake --build --preset=full

# Custom configuration
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DKPU_BUILD_PYTHON_BINDINGS=ON \
  -DKPU_ENABLE_OPENMP=ON \
  -GNinja
cmake --build build
```

#### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `KPU_BUILD_TESTS` | ON | Build test suite |
| `KPU_BUILD_EXAMPLES` | ON | Build examples |
| `KPU_BUILD_TOOLS` | ON | Build development tools |
| `KPU_BUILD_PYTHON_BINDINGS` | ON | Build Python bindings |
| `KPU_BUILD_BENCHMARKS` | ON | Build benchmark suite |
| `KPU_BUILD_MODELS` | ON | Build architecture models |
| `KPU_BUILD_DOCS` | OFF | Build documentation |
| `KPU_ENABLE_OPENMP` | ON | Enable OpenMP parallelization |
| `KPU_ENABLE_CUDA` | OFF | Enable CUDA support |
| `KPU_ENABLE_OPENCL` | OFF | Enable OpenCL support |
| `KPU_ENABLE_PROFILING` | OFF | Enable profiling support |
| `KPU_ENABLE_SANITIZERS` | OFF | Enable sanitizers (debug builds) |

### Install Python Package

```bash
# After building with Python bindings enabled
pip3 install -e .

# Or use the Python module directly from build directory
export PYTHONPATH=$PWD/build:$PYTHONPATH
```

## domain_flow Integration

The simulator integrates with the [domain_flow](https://github.com/branes-ai/domain_flow) intermediate representation for computational graphs.

### Build with domain_flow

```bash
# Option 1: Local installation (recommended for development)
cmake --preset=release -DKPU_DOMAIN_FLOW_LOCAL_PATH=~/dev/domain_flow
cmake --build --preset=release

# Option 2: FetchContent (automatic download - requires CMake 3.28+)
cmake --preset=release  # Automatically fetches domain_flow from GitHub
cmake --build --preset=release

# Option 3: JSON-only mode (no domain_flow dependency)
cmake --preset=release -DKPU_USE_DOMAIN_FLOW=OFF
cmake --build --preset=release
```

### Load Computational Graphs

```cpp
#include <sw/compiler/graph_loader.hpp>

// Load from domain_flow native format (.dfg)
auto graph = sw::kpu::compiler::load_graph("models/mobilenet_v1.dfg");

// Or load from JSON format
auto graph = sw::kpu::compiler::load_graph("models/simple_matmul.json");

// Inspect graph
std::cout << "Graph: " << graph->name << "\n";
std::cout << "Operators: " << graph->operators.size() << "\n";
std::cout << "Tensors: " << graph->tensors.size() << "\n";

// Validate graph structure
if (graph->validate()) {
    std::cout << "✓ Graph is valid\n";
}
```

For more details, see [domain_flow integration guide](docs/domain-flow-integration.md).

## Architecture Overview

The KPU simulator models a specialized hardware accelerator with:

### Core Components

- **Memory Hierarchy**: External memory (1GB), L3 tile, L2 banks, L1 buffers, scratchpad
- **Data Movement Engines**:
  - DMA engine for asynchronous transfers
  - BlockMover for block-level data movement
  - Streamer for stream-based data movement
- **Compute Engines**:
  - ComputeFabric for general-purpose compute
  - SystolicArray (tau111_s001) for matrix multiplication
- **Compiler Infrastructure**: Graph loader, operator mapping, schedule generation (WIP)
- **Configuration System**: JSON-based system configuration
- **Trace Logger**: Performance tracing and analysis

### Key Features

- **Modern C++20**: RAII, smart pointers, concepts, ranges
- **Thread-Safe Design**: All components support concurrent access
- **Comprehensive Testing**: 30/30 tests passing with CTest integration
- **Python Integration**: NumPy-compatible Python API via pybind11
- **Cross-Platform**: Windows, Linux, macOS support

## Usage Examples

### C++ API

#### Basic System Setup

```cpp
#include <sw/system/toplevel.hpp>
#include <sw/kpu/kpu_simulator.hpp>

// Option 1: Create with default configuration
sw::sim::SystemSimulator system;
system.initialize();

// Option 2: Load from JSON configuration file
sw::sim::SystemSimulator system("configs/default_kpu.json");
system.initialize();

// Get KPU instance
auto* kpu = system.get_kpu(0);
```

#### Simple KPU Configuration

```cpp
#include <sw/kpu/kpu_simulator.hpp>

// Create KPU with custom configuration
sw::kpu::KPUSimulator::Config config(
    2,      // 2 memory banks
    1024,   // 1GB each
    100,    // 100 GB/s bandwidth
    2,      // 2 scratchpads
    64,     // 64KB each
    2,      // 2 compute tiles
    2       // 2 DMA engines
);

sw::kpu::KPUSimulator kpu(config);

// Check configuration
std::cout << "Using systolic arrays: " << kpu.is_using_systolic_arrays() << "\n";
std::cout << "Systolic array size: "
          << kpu.get_systolic_array_rows() << "x"
          << kpu.get_systolic_array_cols() << "\n";
```

#### Matrix Multiplication

See [`examples/basic/matrix_multiply.cpp`](examples/basic/matrix_multiply.cpp) for a complete example.

#### Data Movement Pipeline

See [`examples/basic/data_movement_pipeline.cpp`](examples/basic/data_movement_pipeline.cpp) for DMA and data orchestration examples.

### Python API

#### Basic Usage

```python
import stillwater_kpu as kpu
import numpy as np

# Create simulator with context manager
with kpu.Simulator() as sim:
    print(f"Main memory: {sim.main_memory_size // (1024**3)} GB")
    print(f"Scratchpad: {sim.scratchpad_size // (1024**2)} MB")

    # Matrix multiplication
    A = np.random.randn(100, 200).astype(np.float32)
    B = np.random.randn(200, 150).astype(np.float32)

    # KPU computation
    C = sim.matmul(A, B)

    # Verify against NumPy
    C_numpy = A @ B
    assert np.allclose(C, C_numpy), "Results don't match!"
    print("✓ Results match NumPy reference")
```

#### Performance Benchmarking

```python
import stillwater_kpu as kpu

with kpu.Simulator() as sim:
    # Benchmark matrix multiplication
    results = sim.benchmark_matmul(
        M=256, N=256, K=256,
        iterations=10
    )

    print(f"Matrix size: {results['matrix_size']}")
    print(f"KPU time: {results['kpu_time_ms']:.2f} ms")
    print(f"NumPy time: {results['numpy_time_ms']:.2f} ms")
    print(f"KPU GFLOPS: {results['kpu_gflops']:.2f}")
```

#### Advanced Examples

See the [`examples/python/`](examples/python/) directory for:
- Neural network layer computation
- Performance scaling analysis
- Matrix chain optimization
- Educational demonstrations

## Testing

### Run Tests with CTest

```bash
# Run all KPU simulator tests (excludes external domain_flow tests)
cd build
ctest --output-on-failure

# Or use the helper script
./scripts/run_tests.sh

# Run specific test
ctest -R graph_loader -V

# Run with specific preset
ctest --preset=unit          # Unit tests only
ctest --preset=integration   # Integration tests only
ctest --preset=performance   # Performance tests only
```

### Test Categories

The test suite includes 30 comprehensive tests:

| Category | Tests | Coverage |
|----------|-------|----------|
| Memory | 4 | Allocation, sparse memory, memory maps |
| DMA | 6 | Basic transfers, performance, tensor movement, tracing |
| Data Movement | 4 | BlockMover, Streamer operations |
| Compute | 3 | ComputeFabric, SystolicArray |
| Storage Scheduler | 5 | EDDO/IDDO workflows, performance |
| System | 3 | Configuration, formatting |
| Integration | 3 | End-to-end, multi-component, Python bindings |
| Compiler | 1 | Graph loader |

**Status: 30/30 PASSING ✅**

### Excluding External Tests

The build system includes tests from external dependencies (domain_flow). To run only KPU tests:

```bash
# Recommended: Exclude external tests
ctest --test-dir build -E "^(dsp_|nla_|dfa_|dnn_|ctl_|cnn_)" --output-on-failure
```

## Project Structure

```
KPU-simulator/
├── include/sw/              # Public C++ headers
│   ├── system/              # System simulator (toplevel, config)
│   ├── kpu/                 # KPU components
│   ├── memory/              # Memory hierarchy
│   ├── compute/             # Compute engines
│   ├── datamovement/        # Data movement (DMA)
│   ├── compiler/            # Graph loader & compiler
│   ├── driver/              # Memory manager
│   └── trace/               # Tracing infrastructure
├── src/                     # Implementation
│   ├── system/              # System implementation
│   ├── components/          # Component implementations
│   ├── compiler/            # Graph loader implementation
│   ├── bindings/            # C and Python bindings
│   ├── simulator/           # Core simulator
│   └── driver/              # Driver implementation
├── tests/                   # Test suite (30 tests)
├── examples/                # C++ and Python examples
│   ├── basic/               # Basic C++ examples
│   └── python/              # Python examples
├── docs/                    # Documentation (50+ files)
├── cmake/                   # CMake modules
├── scripts/                 # Build and utility scripts
├── configs/                 # Configuration files
├── test_graphs/             # Test computational graphs
└── CMakePresets.json        # CMake presets configuration
```

## Performance Characteristics

### Compute Performance

- **Algorithm**: Standard GEMM (General Matrix Multiply) with systolic array support
- **Parallelization**: OpenMP for matrices > 1024 elements
- **Precision**: Single-precision floating-point (float32)
- **Memory**: Optimized cache-friendly access patterns

### Memory Hierarchy

- **External Memory**: 1GB (configurable)
- **L3 Tile**: Main working memory
- **L2 Banks**: Mid-level cache
- **L1 Buffers**: Fast scratch memory
- **Scratchpad**: Software-managed (1MB default)

### Data Movement

- **DMA Engine**: Asynchronous transfers with address-based API
- **BlockMover**: Efficient block-level data movement
- **Streamer**: Stream-based data orchestration

## API Reference

### C++ Classes

#### `sw::sim::SystemSimulator`
Top-level system simulator that manages all components.

**Methods:**
- `SystemSimulator()` - Create with default configuration
- `SystemSimulator(const SystemConfig& config)` - Create with specific configuration
- `SystemSimulator(const std::filesystem::path& config_file)` - Load from JSON
- `bool initialize()` - Initialize simulator
- `bool is_initialized() const` - Check initialization status
- `sw::kpu::KPUSimulator* get_kpu(size_t index)` - Get KPU by index
- `void print_config() const` - Print configuration summary
- `void shutdown()` - Cleanup resources

#### `sw::kpu::KPUSimulator`
KPU accelerator simulator.

**Configuration:**
```cpp
struct Config {
    size_t memory_bank_count;
    size_t memory_bank_size_mb;
    double memory_bandwidth_gbps;
    size_t scratchpad_count;
    size_t scratchpad_size_kb;
    size_t compute_tile_count;
    size_t dma_engine_count;
};
```

**Methods:**
- `KPUSimulator(const Config& config)` - Create with configuration
- `bool is_using_systolic_arrays() const` - Check systolic array usage
- `size_t get_systolic_array_rows() const` - Get systolic array dimensions
- `size_t get_systolic_array_cols() const` - Get systolic array dimensions

#### `sw::kpu::compiler::GraphLoader`
Computational graph loader.

**Functions:**
- `std::unique_ptr<ComputationalGraph> load_graph(const std::string& path)` - Load graph from .dfg or .json

### Python Classes

#### `stillwater_kpu.Simulator`
Python wrapper for KPU simulator.

**Methods:**
- `__init__(main_memory_size=1<<30, scratchpad_size=1<<20)` - Create simulator
- `matmul(A, B)` - Matrix multiplication (NumPy arrays)
- `benchmark_matmul(M, N, K, iterations=10)` - Performance benchmark
- `__enter__() / __exit__()` - Context manager support

**Properties:**
- `main_memory_size` - Main memory size in bytes
- `scratchpad_size` - Scratchpad size in bytes

## Error Handling

### C++ Exceptions
- `std::out_of_range` - Memory access violations
- `std::invalid_argument` - Invalid parameters
- `std::runtime_error` - Resource allocation failures

### Python Exceptions
- `KPUMemoryError` - Memory access errors
- `KPUDimensionError` - Matrix dimension mismatches
- `KPUError` - General simulator errors

## Development Status

**Current Version: 0.1.0 (Beta)**

### Completed ✅
- Core simulator architecture
- Memory hierarchy implementation
- DMA and data movement engines
- Compute fabric and systolic arrays
- Python bindings with NumPy integration
- Comprehensive test suite (30/30 passing)
- domain_flow graph loading
- Configuration system
- Trace logging

### In Progress 🚧
- Schedule generation from computational graphs
- Tensor metadata extraction
- Framework importers (ONNX, PyTorch, JAX)
- Optimization passes

### Planned 📋
- Additional data types (int8, int16, bfloat16)
- Advanced operations (convolution, activation functions)
- Real-time performance monitoring UI
- Multi-KPU distributed simulation

## Documentation

- **[Developer Setup Guide](README_dev.md)** - Development environment setup
- **[Quick Start Guide](QUICK_START.md)** - domain_flow integration quick start
- **[Architecture Specification](docs/kpu_architecture.md)** - Detailed KPU architecture
- **[domain_flow Integration](docs/domain-flow-integration.md)** - Computational graph integration
- **[Configuration Guide](docs/configuration-architecture.md)** - System configuration
- **[Data Orchestration](docs/data-orchestration.md)** - Data movement details
- **[Tracing System](docs/tracing-system.md)** - Performance tracing

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Code Style**: Follow modern C++20 best practices
2. **Thread Safety**: Maintain thread-safe design for shared components
3. **Testing**: Add comprehensive tests for new features (use CTest)
4. **Documentation**: Update documentation for API changes
5. **Performance**: Benchmark performance impact of modifications

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature

# Build with debug preset
cmake --preset=debug
cmake --build --preset=debug

# Run tests
ctest --preset=default --output-on-failure

# Submit pull request
```

# Testing Guide

## Running Tests

### Run All KPU-Simulator Tests (Recommended)

Exclude domain_flow's own tests which may fail independently:

```bash
# From project root
ctest --test-dir build -E "^(dsp_|nla_|dfa_|dnn_|ctl_|cnn_)" --output-on-failure

# Or from build directory
cd build
ctest -E "^(dsp_|nla_|dfa_|dnn_|ctl_|cnn_)" --output-on-failure
```

**Result**: 30/30 tests pass ✅

### Run All Tests (Including domain_flow)

```bash
ctest --test-dir build --output-on-failure
```

**Note**: This will include 12 domain_flow tests (tests #1-12) which may fail. These failures are from the external domain_flow library and do not affect KPU-simulator functionality.

### Run Specific Test Suites

```bash
# Memory tests only
ctest --test-dir build -R "memory" -V

# DMA tests only
ctest --test-dir build -R "dma" -V

# Graph loader tests
ctest --test-dir build -R "graph_loader" -V

# Storage scheduler tests
ctest --test-dir build -R "storage" -V

# Integration tests
ctest --test-dir build -R "integration" -V
```

### Run Single Test

```bash
ctest --test-dir build -R "test_name" -V
```

## Test Categories

### KPU-Simulator Tests (30 tests)
- **System Tests**: Configuration, formatting
- **Memory Tests**: Allocation, sparse memory, memory map
- **DMA Tests**: Basic, performance, tensor movement, tracing
- **Block Mover Tests**: Basic operations, tracing
- **Streamer Tests**: Basic operations, tracing
- **Compute Tests**: Basic fabric operations, systolic array
- **Storage Scheduler Tests**: IDDO, EDDO workflows, performance
- **Integration Tests**: End-to-end, multi-component, Python bindings

### Domain Flow Tests (12 tests - external)
These are from the domain_flow library dependency:
- Tests #1-12: dsp_, nla_, dfa_, dnn_, ctl_, cnn_

**Note**: These tests may fail or not run properly. They test domain_flow functionality, not KPU-simulator.

## CI/CD Integration

### GitHub Actions

Add to `.github/workflows/*.yml`:

```yaml
- name: Run tests
  run: |
    ctest --test-dir build -E "^(dsp_|nla_|dfa_|dnn_|ctl_|cnn_)" --output-on-failure
```

Or use the exclude pattern file:

```yaml
- name: Run tests
  run: |
    ctest --test-dir build -E "$(cat .github/workflows/test-exclude-pattern.txt)" --output-on-failure
```

## Test Results Summary

```bash
# Expected results when excluding domain_flow tests:
100% tests passed, 0 tests failed out of 30

Total Test time (real) =  ~15 sec
```

## Troubleshooting

### All Tests Fail
- Check build succeeded: `cmake --build build`
- Verify working directory: run from project root or use `--test-dir build`

### Graph Loader Tests Skip
- Run: `scripts/copy_domain_flow_graphs.sh`
- This copies .dfg test files from domain_flow dependency

### Python Tests Fail
- Verify Python bindings built: check for `stillwater_kpu.*.so` in build output
- Check Python environment matches build (Python 3.12 expected)

### Memory Tests Fail
- May need larger system memory
- Some tests validate sparse memory allocation

## Performance Benchmarks

Some tests include performance benchmarks:
- `dma_performance_test`: DMA throughput
- `storage_scheduler_performance_test`: EDDO command processing
- `end_to_end`: Full system performance

Run with:
```bash
ctest --test-dir build -R "performance" -V
```

## License

This project is released under the MIT License. See LICENSE file for details.

---

**Stillwater Computing, Inc.**
*Accelerating Innovation (TM)*

**Version:** 0.1.0
**Build System:** CMake 3.20+ with presets
**Language:** C++20
**Python Support:** Python 3.8-3.12
