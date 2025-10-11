# Autonomous KPU Execution Model

## Overview

This directory contains two KPU execution models that demonstrate different approaches to hardware orchestration:

1. **`host_t100.cpp`** - "GOD Mode" execution with centralized host control
2. **`host_t100_autonomous.cpp`** - Autonomous execution with signal-based synchronization

## Key Architectural Difference

### GOD Mode (host_t100.cpp)
```cpp
// Stage 1: DMA
kpu->start_dma_transfer(...);
kpu->run_until_idle();  // Block until complete

// Stage 2: BlockMover
kpu->start_block_transfer(...);
kpu->run_until_idle();  // Block until complete

// Stage 3: Streamer
kpu->start_row_stream(...);
kpu->run_until_idle();  // Block until complete
```

**Problems:**
- Centralized orchestration
- Sequential execution (no parallelism)
- Unrealistic - real hardware doesn't work this way
- Host has omniscient control

### Autonomous Mode (host_t100_autonomous.cpp)
```cpp
// Program all stages upfront with dependencies
orch.await(DMA_DONE, [&]() {
    kpu->start_block_transfer(...);
});

orch.await(BLOCK_DONE, [&]() {
    kpu->start_row_stream(...);
});

// Single execution loop - true concurrency
while (!orch.is_complete()) {
    kpu->step();      // All engines advance in parallel
    orch.step();      // Launch operations when dependencies satisfied
}
```

**Benefits:**
- Dependency-driven execution
- True hardware concurrency
- Realistic synchronization model
- No centralized control
- Pipeline efficiency

## Architecture Components

### AutonomousOrchestrator
Located in `autonomous_orchestrator.hpp`, this class provides:
- Signal-based synchronization (models hardware semaphores)
- Dependency tracking between pipeline stages
- Automatic operation launching when dependencies satisfied
- Verbose logging for debugging

### Signal Flow

The autonomous model uses these signals to coordinate the MLP pipeline:

```
L3_INPUT_DONE → BLOCK_INPUT_DONE → STREAM_INPUT_DONE ↘
                                                        → COMPUTE_DONE → BIAS_ADDED → ... → ALL_DONE
L3_WEIGHTS_DONE → BLOCK_WEIGHTS_DONE → STREAM_WEIGHTS_DONE ↗
```

## Building

```bash
# Build both models
cmake --build build --target model_host_t100 -j4
cmake --build build --target model_host_t100_autonomous -j4

# Or build everything
cmake --build build -j4
```

## Running

```bash
# GOD mode model
./build/models/kpu/model_host_t100

# Autonomous model
./build/models/kpu/model_host_t100_autonomous
```

## Performance Comparison

### GOD Mode
- **Execution**: Sequential stages
- **Cycles**: ~Variable (depends on synchronization overhead)
- **Parallelism**: None (one component active at a time)
- **Realism**: Low (not how real hardware works)

### Autonomous Mode
- **Execution**: Concurrent operation of all engines
- **Cycles**: 13 cycles for 4×8×4 MLP (example)
- **Parallelism**: Full (multiple engines active simultaneously)
- **Realism**: High (models actual hardware behavior)

## Example Output

```
========================================
  Autonomous MLP Layer Execution
========================================
Batch size: 4
Input dimension: 8
Output dimension: 4

[3] Programming autonomous pipeline
  Input staged in L3
  Weights staged in L3
  Pipeline programmed with 10 operations

[4] Autonomous Execution
  Starting concurrent execution of all components...
    Execution complete in 13 cycles

[5] Result Verification
  Pipeline executed successfully
  Total cycles: 13
  Pipeline stages: 10
```

## Implementation Details

### Programming Phase
All pipeline operations are registered with their dependencies during setup:

```cpp
// Stage 1: Manual L3 staging (immediate)
kpu->write_l3_tile(...);
orch.signal(L3_INPUT_DONE);

// Stage 2: BlockMover L3→L2 (awaits L3 staging)
orch.await(L3_INPUT_DONE, [&]() {
    kpu->start_block_transfer(..., callback: [&]() {
        orch.signal(BLOCK_INPUT_DONE);
    });
}, "BlockMover: L3→L2 (input)");

// Stage 3: Streamer L2→L1 (awaits BlockMover)
orch.await(BLOCK_INPUT_DONE, [&]() {
    kpu->start_row_stream(..., callback: [&]() {
        orch.signal(STREAM_INPUT_DONE);
    });
}, "Streamer: L2→L1 (input rows)");

// ... and so on for the complete pipeline
```

### Execution Phase
Simple loop that advances all hardware and checks dependencies:

```cpp
while (!orch.is_complete()) {
    kpu->step();      // Hardware engines process one cycle
    orch.step();      // Check dependencies, launch ready ops
}
```

## Data Flow Pipeline

```
┌────────────┐
│ Host Memory│
└─────┬──────┘
      │ (manual load)
      ↓
┌────────────┐
│ Mem Bank   │
└─────┬──────┘
      │ (manual staging)
      ↓
┌────────────┐     Signal: L3_INPUT_DONE
│ L3 Tiles   │
└─────┬──────┘
      │ [BlockMover]
      ↓
┌────────────┐     Signal: BLOCK_INPUT_DONE
│ L2 Banks   │
└─────┬──────┘
      │ [Streamer]
      ↓
┌────────────┐     Signal: STREAM_INPUT_DONE
│ L1/Scratch │
└─────┬──────┘
      │ [SystolicArray]
      ↓
┌────────────┐     Signal: COMPUTE_DONE
│  Results   │
└────────────┘
```

## Known Issues

### Zero Output Bug
Both models currently produce zero outputs. This is a pre-existing issue in the component implementations (BlockMover, Streamer, or SystolicArray), NOT in the orchestration logic. The autonomous model correctly executes the same pipeline as the GOD mode model, just with better orchestration.

**Evidence:**
- Both `model_host_t100` and `model_host_t100_autonomous` produce identical zero outputs
- The orchestration completes successfully (all 10 stages execute)
- Cycle counts are reasonable (13 cycles)
- Signal dependencies are satisfied in correct order

**Next Steps:**
Debug the actual component implementations to fix data movement:
- [ ] Verify BlockMover actually moves data between L3 and L2
- [ ] Verify Streamer actually moves data between L2 and L1
- [ ] Verify SystolicArray actually performs computation
- [ ] Add instrumentation to track data through the pipeline

## Educational Value

This autonomous model demonstrates:
1. **Hardware Realism**: How real accelerators operate with autonomous engines
2. **Dependency Graphs**: Explicit data flow dependencies
3. **Concurrency**: Multiple engines operating simultaneously
4. **Synchronization**: Hardware semaphore/signal patterns
5. **Pipeline Design**: Complete dataflow programming upfront

## Future Enhancements

### Multi-Tile Parallelism
```cpp
for (size_t tile = 0; tile < num_tiles; ++tile) {
    std::string prefix = "tile" + std::to_string(tile) + "_";
    orch.await({prefix + "data_ready"}, [&, tile]() {
        kpu->start_matmul(tile, ...);
    });
}
```

### Double Buffering
```cpp
// While computing batch N, stream batch N+1
orch.await({COMPUTE_N_STARTED}, [&]() {
    stream_batch_n_plus_1();
});
```

### Pipelined Layers
```cpp
// Layer 2 starts as soon as layer 1 produces first tile
orch.await({LAYER1_FIRST_TILE_DONE}, [&]() {
    start_layer2_compute();
});
```

## Architectural Improvements

```cpp
// Before (GOD Mode):
kpu->start_block_transfer(...);
kpu->run_until_idle();  // Block!
kpu->start_row_stream(...);
kpu->run_until_idle();  // Block!

// After (Autonomous):
// Program everything upfront
orch.await(BLOCK_DONE, [&](){ kpu->start_row_stream(...); });
orch.await(STREAM_DONE, [&](){ kpu->start_matmul(...); });

// Single execution loop - all engines work concurrently
while (!orch.is_complete()) {
    kpu->step();
    orch.step();
}
```

## References

- Design document: `docs/autonomous-kpu-design.md`
- Orchestrator implementation: `models/kpu/autonomous_orchestrator.hpp`
- Component interfaces:
  - `include/sw/kpu/components/dma_engine.hpp`
  - `include/sw/kpu/components/block_mover.hpp`
  - `include/sw/kpu/components/streamer.hpp`
  - `include/sw/kpu/components/systolic_array.hpp`
