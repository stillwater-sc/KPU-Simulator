# Changelog

All notable changes to the KPU Simulator project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - 2025-11-26
- **Domain Flow Execution (DFX) Layer**
  - Created PTX-equivalent hardware-agnostic intermediate representation for KPU
  - `include/sw/compiler/dfx/dfx.hpp` - Core DFX types and structures:
    - `DataType`, `MemoryLevel`, `DataflowStrategy` enums
    - `TensorDescriptor`, `TileSpec`, `TilingConfig` structures
    - `Operation` base class with `DataMoveOp`, `ComputeOp`, `BarrierOp` derived types
    - `Program` struct containing complete compiled kernel representation
  - `include/sw/compiler/dfx/dfx_object_file.hpp` - JSON serialization for .kpu files

- **KPU Kernel Compiler (`kpu-kernel-compiler`)**
  - Full compilation pipeline from DFG to .kpu object files
  - `tools/compiler/kpu-kernel-compiler/dfg_parser.hpp/cpp` - DFG/JSON file parsing
  - `tools/compiler/kpu-kernel-compiler/dfx_generator.hpp/cpp` - DFX program generation
  - `tools/compiler/kpu-kernel-compiler/object_writer.hpp/cpp` - .kpu file writer
  - CLI options: `-o`, `-d` (dataflow), `-t` (tile-strategy), `--emit-dfx`, `--dump`, `-v`
  - Supports output-stationary, weight-stationary, and input-stationary dataflows
  - Integrates with existing TileOptimizer for optimal tile size selection

- **KPU Loader Framework** (skeleton)
  - `tools/runtime/kpu-loader/` - Loader/driver framework
  - `object_reader.hpp/cpp` - Read and validate .kpu files
  - `schedule_binder.hpp/cpp` - Bind DFX operations to concrete hardware resources
  - Maps abstract operations to DMA engines, BlockMovers, and Streamers

- **Tools Directory Reorganization**
  - New category-based structure: `compiler/`, `runtime/`, `analysis/`, `development/`, `configuration/`, `benchmark/`
  - `kpu_add_tool()` CMake helper function for consistent tool creation
  - Moved Python tools to appropriate subdirectories

- **Implementation Plan Document**
  - `docs/compiler/KPU_COMPILER_IMPLEMENTATION_PLAN.md` - Comprehensive design document
  - Covers architecture, DFX format, object file structure, CLI design

### Changed - 2025-11-26
- **Renamed KIR to DFX**
  - Renamed namespace from `sw::kpu::compiler::kir` to `sw::kpu::compiler::dfx`
  - Renamed directory from `include/sw/compiler/kir/` to `include/sw/compiler/dfx/`
  - Renamed files: `kir.hpp` → `dfx.hpp`, `object_file.hpp` → `dfx_object_file.hpp`
  - Renamed class: `KIRGenerator` → `DFXGenerator` (with backward compatibility alias)
  - Updated version constants: `KIR_VERSION_*` → `DFX_VERSION_*`
  - Updated CLI flag: `--emit-kir` → `--emit-dfx`
  - Updated JSON key: `"kir_version"` → `"dfx_version"`

### Added - 2025-11-25
- **Strategy-Aware L2/L3 Scheduling**
  - Implemented proper dataflow strategy loop ordering in L2 tile scheduler
  - Added strategy-aware execution in L3 scheduler
  - Strategies now produce different (and correct) overfetch results:
    - **WS (Weight-Stationary)**: `tk → ti → tj` keeps B tiles resident
    - **IS (Input-Stationary)**: `tk → tj → ti` keeps A tiles resident
    - **OS (Output-Stationary)**: `ti → tj → tk` keeps C tiles resident
  - Added `strategy` field to `L2Schedule` struct to propagate strategy choice

- **Distributed L3 Support in Analysis Tools**
  - Added 1MB and 2MB L3 sizes to focused analysis (3→5 sizes, 108→180 configs)
  - Added 1MB and 2MB L3 sizes to comprehensive analysis (5→7 sizes, 405→567 configs)
  - Created `run_comprehensive_overnight.sh` convenience script

- **Analysis Documentation**
  - Created `L3_ANALYSIS_UPDATED.md` documenting distributed L3 support
  - Created `STRATEGY_AWARE_SCHEDULING_RESULTS.md` documenting bug fix and results
  - Updated analysis tools to use strategy-aware scheduling

### Fixed - 2025-11-25
- **Critical Overfetch Asymmetry Bug**
  - Fixed L2 scheduler's `generate_compute_order()` ignoring strategy parameter
  - Fixed L3 scheduler's `simulate_l2_execution()` using hard-coded OS loops
  - **Impact**: 380× improvement for 32k×7k workload (34.56× → 0.90× with WS)
  - Tall and wide matrices now show proper symmetry with correct strategy selection

- **Compiler Warnings**
  - Fixed unused parameter warnings in `l3_overfetch_analyzer.cpp`
  - Fixed unused parameter warnings in `schedule_characterizer_demo.cpp`

### Changed - 2025-11-25
- **L2 Tile Scheduler**
  - Moved `ReplacementPolicy` and `SchedulingStrategy` enums before `L2Schedule` struct
  - Updated `generate_compute_order()` to respect strategy parameter
  - Strategy now stored in generated L2 schedules

- **L3 Analysis Tools**
  - `l3_focused_analysis.cpp` generates separate L2 schedules for each strategy
  - `l3_comprehensive_analysis.cpp` applies strategy-aware scheduling
  - Both tools now test 1MB and 2MB L3 configurations

### Added - 2025-11-23
- **Tile Notation Improvements** in `ScheduleGenerator`
  - Added `TileIndex::label_A()`, `label_B()`, `label_C()` methods for proper mathematical notation
  - Tile labels now show correct dimensionality:
    - `A_tile[ti,tk]` - A tile indexed by M-dimension and K-dimension
    - `B_tile[tk,tj]` - B tile indexed by K-dimension and N-dimension
    - `C_tile[ti,tj]` - C tile indexed by M-dimension and N-dimension
  - Kept legacy `label(char)` method for backwards compatibility

- **Double-Buffering Infrastructure** in `ScheduleGenerator`
  - Implemented `apply_double_buffering()` method
  - Buffer ID tracking for commands (alternates between 0 and 1)
  - Dependency adjustment for buffer switching
  - **Known Issue**: Does not properly model resource constraints

- **Pipelining Infrastructure** in `ScheduleGenerator`
  - Implemented `apply_pipelining()` method
  - Dependency refinement to enable parallelism
  - **Known Issue**: Shows physically impossible parallelism (multiple commands on same resource)

- **Enhanced Timing Estimation** in `ScheduleGenerator`
  - Improved `estimate_timing()` to handle parallel command execution
  - Proper dependency-based scheduling
  - Commands scheduled when all dependencies satisfied

- **Command Timeline Visualization** in `schedule_generator_demo`
  - Added detailed timeline printing in `compare_strategies()`
  - Shows all commands with start/end cycles, duration, and buffer IDs
  - Changed demo matrix size from 512×512×512 to 128×128×128 for readable output
  - Visual comparison of Sequential, Double-buffered, and Fully-pipelined strategies

- **Session Documentation**
  - Created `docs/sessions/` directory for session logs
  - Added comprehensive session log for 2025-11-23 pipelining work

### Changed - 2025-11-23
- **ScheduleGenerator** tile label generation
  - Updated all command generation to use new tile notation
  - `generate_dma_commands()`, `generate_block_move_commands()`, `generate_stream_commands()`, `generate_compute_commands()` now use `TileIndex::label_A/B/C()`

- **schedule_generator_demo.cpp**
  - `compare_strategies()` now prints full command timeline for all three strategies
  - Matrix size reduced to 128×128×128 for strategy comparison (from 512×512×512)
  - Added detailed explanations of pipelining benefits

### Fixed - 2025-11-23
- **Compilation Error** in `schedule_generator.cpp`
  - Added missing `#include <iostream>` header

### Known Issues - 2025-11-23

#### Critical Design Flaws in Pipelining Implementation

The current pipelining and double-buffering implementation has fundamental flaws:

1. **Resource Constraints Not Modeled**
   - Schedules show physically impossible parallelism (e.g., 16 BlockMoves starting simultaneously)
   - No modeling of finite resource capacity (DMA engines, BlockMovers, Streamers)
   - No resource allocation or scheduling logic
   - **Impact**: Generated schedules cannot execute on actual hardware

2. **No True Overlap**
   - Dependencies don't correctly model producer-consumer relationships across pipeline stages
   - No real overlap between data movement and compute despite "pipelined" strategy
   - **Impact**: Performance estimates are incorrect

3. **Improper Tile Reuse**
   - Doesn't model tile reuse across K-dimension
   - Treats reused tiles as independent loads
   - **Impact**: Overstates memory traffic, incorrect cache modeling

4. **Missing Constraints**
   - No spatial routing constraints (which L3 tile connects to which L2 bank)
   - No bandwidth modeling for interconnects
   - No systolic array scheduling
   - **Impact**: Schedules violate physical hardware constraints

#### Test Coverage Gaps

- All 32 tests in `test_schedule_generator.cpp` pass
- **However**: Tests don't validate:
  - Resource constraint satisfaction
  - Physical feasibility of parallelism
  - Correct tile reuse modeling
  - Actual data movement and compute overlap

#### Recommendations for Future Work

See `docs/sessions/2025-11-23_schedule_generator_pipelining.md` for detailed recommendations:
- Phase 1: Add explicit resource capacity modeling and resource scheduler
- Phase 2: Model network topology and spatial constraints
- Phase 3: Implement tile reuse optimization
- Phase 4: Add bandwidth modeling for interconnects
- Phase 5: Correct dependency graph with resource hazards
- Alternative: Consider polyhedral scheduling approach (MLIR, Halide, TVM)

### Testing - 2025-11-23
- ✅ All 32 tests in `test_schedule_generator` pass
- ✅ Clean build with no warnings
- ✅ Demo executable runs and produces output
- ⚠️  Output shows physically impossible parallelism (design flaw, not implementation bug)

---

## Notes

### Session Logs
Detailed session logs are maintained in `docs/sessions/` directory:
- `2025-11-26_dfx_compiler_implementation.md` - DFX layer and kernel compiler implementation
- `2025-11-25_strategy_aware_scheduling.md` - Strategy-aware L2/L3 scheduling fix
- `2025-11-23_schedule_generator_pipelining.md` - Double-buffering and pipelining attempt

### Version History
This CHANGELOG was created on 2025-11-23 to track changes going forward.
Previous changes to the KPU simulator are documented in:
- Git commit history
- Session logs in `docs/sessions/`
- Documentation in `docs/` directory
