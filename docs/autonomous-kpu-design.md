# Autonomous KPU Data Flow Model Design

## Executive Summary

This document outlines the design for creating a more realistic and accurate autonomous execution model for the KPU simulator. The current `host_t100.cpp` implementation uses a "GOD mode" approach where the host has omniscient control over all hardware components. The new `host_t100_autonomous.cpp` will model how real hardware actually works: autonomous engines executing concurrently with explicit synchronization primitives.

## Current "GOD Mode" Problems

The current implementation in `execute_mlp_layer()` (host_t100.cpp:260-448) has these issues:

### 1. Centralized Orchestration
The host explicitly manages every data movement step, acting as a central controller that real hardware doesn't have.

### 2. Synchronous Execution
After each operation, the code calls `run_until_idle()`:
- Line 349: After BlockMover input transfer
- Line 356: After BlockMover weight transfer
- Line 373: After Streamer input streaming
- Line 380: After Streamer weight streaming
- Line 396: After matrix multiplication
- Line 418: After result streaming back
- Line 426: After BlockMover result transfer

### 3. No Inter-Component Collaboration
Components don't signal each other; the host acts as intermediary between all stages.

### 4. Manual Memory Staging
Direct read/write operations for L3 transfers (lines 328-335):
```cpp
kpu->read_memory_bank(bank_id, input_addr, temp_buffer.data(), input.size() * sizeof(float));
kpu->write_l3_tile(l3_tile_id, l3_input_addr, temp_buffer.data(), input.size() * sizeof(float));
```

### 5. Omniscient Control
The host has complete visibility and control over all timing, which doesn't reflect real hardware constraints.

## Real Hardware Behavior

### Data Flow Pipeline
```
Host Memory → [DMA] → L3 Tiles → [BlockMover] → L2 Banks → [Streamer] → L1/Scratchpad → [SystolicArray]
```

### Autonomous Execution Characteristics

**Each engine (DMA, BlockMover, Streamer, SystolicArray) is:**
- An **autonomous process** that executes independently
- **Programmed** with a sequence of operations (via enqueue methods)
- **Collaborative** through synchronization primitives (push/await)
- **Concurrent** - no centralized orchestration required

## Existing Component Infrastructure

Good news: The components already have most of what we need!

### DMAEngine (dma_engine.hpp)
- Queue-based operation: `enqueue_transfer()`
- Async processing: `process_transfers()`
- Completion callbacks
- Status checking: `is_busy()`

### BlockMover (block_mover.hpp)
- Queue-based operation: `enqueue_block_transfer()`
- Async processing: `process_transfers()`
- Transformation support (IDENTITY, TRANSPOSE, etc.)
- Completion callbacks
- Status checking: `is_busy()`

### Streamer (streamer.hpp)
- Queue-based operation: `enqueue_stream()`
- Cycle-accurate execution: `update(current_cycle, ...)`
- Bidirectional streaming (L2↔L1)
- Staggering logic for systolic arrays
- Completion callbacks
- Status checking: `is_busy()`, `is_streaming()`

### SystolicArray (systolic_array.hpp)
- Async start: `start_matmul()`
- Cycle-accurate execution: `update(current_cycle, ...)`
- Completion callbacks
- Status checking: `is_busy()`

### StorageScheduler (storage_scheduler.hpp)
- **Dependency management**: Commands with dependencies
- **Barrier operations**: Synchronization points
- **Sequence IDs**: For ordering operations

## What's Missing for True Autonomy

### 1. Synchronization Primitives Between Engines
- **Need**: Push/Await mechanism so engines can signal each other
- **Example**: BlockMover awaits DMA completion, Streamer awaits BlockMover completion
- **Current State**: Completion callbacks exist but don't directly trigger dependent engines

### 2. Dependency Graph for Data Movement
- **Need**: Way to express "Operation B depends on Operation A completing"
- **Similar to**: `StorageScheduler::StorageCommand.dependencies`
- **Current State**: No cross-engine dependency tracking

### 3. Concurrent Execution Framework
- **Need**: All engines execute in parallel, checking dependencies each cycle
- **Current State**: `step()` updates all engines, but there's no dependency-driven execution

### 4. Event/Signal System
- **Need**: Engines can signal completion to waiting engines
- **Current State**: Callbacks exist but are host-facing, not engine-facing

## Implementation Options

### Option 1: Lightweight - Use Completion Callbacks for Chaining

**Approach**: Modify the existing callback system to trigger dependent engine operations

**Example**:
```cpp
// DMA completion triggers BlockMover
dma.enqueue_transfer(..., callback: [&](){
    block_mover.enqueue_transfer(...);
});
```

**Pros**:
- Minimal changes required
- Works with existing infrastructure
- Quick to implement

**Cons**:
- Still somewhat sequential in nature
- Not true concurrent programming model
- Callbacks can become deeply nested (callback hell)
- Hard to visualize data flow dependencies

### Option 2: Medium - Add Event/Signal System (RECOMMENDED)

**Approach**: Create a simple event-based signal system

**Example**:
```cpp
enum class Signal {
    DMA_INPUT_DONE,
    DMA_WEIGHTS_DONE,
    BLOCK_INPUT_DONE,
    BLOCK_WEIGHTS_DONE,
    STREAM_INPUT_DONE,
    STREAM_WEIGHTS_DONE,
    COMPUTE_DONE,
    READBACK_DONE
};

class SignalManager {
    std::unordered_map<Signal, bool> signals;
    std::vector<std::function<void()>> pending_operations;

public:
    void signal(Signal s);
    void wait_for(Signal s, std::function<void()> operation);
    bool is_signaled(Signal s) const;
    void step();  // Check all pending operations, execute if dependencies satisfied
};
```

**Usage Pattern**:
```cpp
// Programming Phase
signal_mgr.wait_for(DMA_INPUT_DONE, [&](){
    // This executes automatically when DMA_INPUT_DONE is signaled
    block_mover.enqueue_transfer(...);
});

// Later, when DMA completes
dma.enqueue_transfer(..., callback: [&](){
    signal_mgr.signal(DMA_INPUT_DONE);
});

// Execution Phase
while (!all_work_complete) {
    kpu->step();           // Advance hardware one cycle
    signal_mgr.step();     // Process any newly satisfied dependencies
}
```

**Pros**:
- Clean abstraction of hardware synchronization
- Enables true dependency expression
- Models real hardware semaphores/flags
- Easy to visualize data flow
- Educational value - shows concurrent nature clearly

**Cons**:
- Requires adding signal management infrastructure
- Components need minor modifications to use signals

### Option 3: Full - Integrate StorageScheduler Pattern

**Approach**: Adopt the dependency graph approach from `StorageScheduler` across all components

**Example**:
```cpp
struct DataMovementCommand {
    size_t sequence_id;
    std::vector<size_t> dependencies;  // Other sequence_ids to wait for
    ComponentType target_component;     // DMA, BlockMover, Streamer, etc.
    std::function<void()> operation;
    std::function<void()> completion_callback;
};

class GlobalScheduler {
    std::unordered_map<size_t, DataMovementCommand> commands;
    std::unordered_map<size_t, bool> completed;

    bool can_execute(const DataMovementCommand& cmd);
    void execute_ready_commands();
};
```

**Pros**:
- Most realistic hardware model
- Supports complex workflows with multiple dependencies
- Natural representation of DAG (Directed Acyclic Graph) execution
- Maximum flexibility

**Cons**:
- Significant refactoring required
- Steeper learning curve
- May be overkill for simple pipelines
- More complex to debug

## Recommended Approach: Option 2

For `host_t100_autonomous.cpp`, we recommend **Option 2** (Event/Signal System) because:

### 1. Realistic
Models how real hardware synchronization works through hardware semaphores/flags

### 2. Manageable Scope
Doesn't require major refactoring of existing components - just adds a coordination layer

### 3. Educational Value
Clearly shows the autonomous, concurrent nature of hardware execution

### 4. Extensible
Can later evolve into Option 3 if needed for more complex scenarios

### 5. Clear Semantics
Signal names make data dependencies explicit and self-documenting

## Proposed Architecture for host_t100_autonomous.cpp

### Signal Manager Implementation

```cpp
class AutonomousOrchestrator {
private:
    struct PendingOperation {
        std::vector<std::string> required_signals;
        std::function<void()> operation;
        bool executed;
    };

    std::unordered_map<std::string, bool> signals;
    std::vector<PendingOperation> pending_operations;

public:
    void signal(const std::string& name) {
        signals[name] = true;
    }

    void await(const std::vector<std::string>& required_signals,
               std::function<void()> operation) {
        pending_operations.push_back({required_signals, operation, false});
    }

    void step() {
        for (auto& op : pending_operations) {
            if (op.executed) continue;

            bool all_satisfied = true;
            for (const auto& sig : op.required_signals) {
                if (!signals[sig]) {
                    all_satisfied = false;
                    break;
                }
            }

            if (all_satisfied) {
                op.operation();
                op.executed = true;
            }
        }
    }

    bool is_complete() const {
        return std::all_of(pending_operations.begin(), pending_operations.end(),
                          [](const auto& op) { return op.executed; });
    }
};
```

### MLP Execution with Autonomous Orchestration

```cpp
bool execute_mlp_layer_autonomous(sw::kpu::KPUSimulator* kpu,
                                   size_t batch_size,
                                   size_t input_dim,
                                   size_t output_dim) {
    AutonomousOrchestrator orch;

    // Define signal names
    const std::string DMA_INPUT_DONE = "dma_input_done";
    const std::string DMA_WEIGHTS_DONE = "dma_weights_done";
    const std::string BLOCK_INPUT_DONE = "block_input_done";
    const std::string BLOCK_WEIGHTS_DONE = "block_weights_done";
    const std::string STREAM_INPUT_DONE = "stream_input_done";
    const std::string STREAM_WEIGHTS_DONE = "stream_weights_done";
    const std::string COMPUTE_DONE = "compute_done";
    const std::string STREAM_OUTPUT_DONE = "stream_output_done";
    const std::string BLOCK_OUTPUT_DONE = "block_output_done";
    const std::string ALL_DONE = "all_done";

    // === Programming Phase: Set up the entire pipeline ===

    // 1. DMA: Memory Bank → L3 (starts immediately)
    kpu->start_dma_transfer(..., callback: [&](){ orch.signal(DMA_INPUT_DONE); });
    kpu->start_dma_transfer(..., callback: [&](){ orch.signal(DMA_WEIGHTS_DONE); });

    // 2. BlockMover: L3 → L2 (waits for DMA)
    orch.await({DMA_INPUT_DONE}, [&](){
        kpu->start_block_transfer(..., callback: [&](){ orch.signal(BLOCK_INPUT_DONE); });
    });
    orch.await({DMA_WEIGHTS_DONE}, [&](){
        kpu->start_block_transfer(..., callback: [&](){ orch.signal(BLOCK_WEIGHTS_DONE); });
    });

    // 3. Streamer: L2 → L1 (waits for BlockMover)
    orch.await({BLOCK_INPUT_DONE}, [&](){
        kpu->start_row_stream(..., callback: [&](){ orch.signal(STREAM_INPUT_DONE); });
    });
    orch.await({BLOCK_WEIGHTS_DONE}, [&](){
        kpu->start_column_stream(..., callback: [&](){ orch.signal(STREAM_WEIGHTS_DONE); });
    });

    // 4. Compute: Systolic Array (waits for both streamers)
    orch.await({STREAM_INPUT_DONE, STREAM_WEIGHTS_DONE}, [&](){
        kpu->start_matmul(..., callback: [&](){ orch.signal(COMPUTE_DONE); });
    });

    // 5. Reverse path for results
    orch.await({COMPUTE_DONE}, [&](){
        kpu->start_row_stream(..., callback: [&](){ orch.signal(STREAM_OUTPUT_DONE); });
    });
    orch.await({STREAM_OUTPUT_DONE}, [&](){
        kpu->start_block_transfer(..., callback: [&](){ orch.signal(BLOCK_OUTPUT_DONE); });
    });
    orch.await({BLOCK_OUTPUT_DONE}, [&](){
        kpu->start_dma_transfer(..., callback: [&](){ orch.signal(ALL_DONE); });
    });

    // === Execution Phase: Run until complete ===
    while (!orch.is_complete()) {
        kpu->step();      // Advance all hardware engines by one cycle
        orch.step();      // Check dependencies, launch ready operations
    }

    return true;
}
```

## Key Differences from GOD Mode

### Before (GOD Mode)
```cpp
// Stage 1: DMA
kpu->start_dma_transfer(...);
kpu->run_until_idle();  // ⛔ Block until complete

// Stage 2: BlockMover
kpu->start_block_transfer(...);
kpu->run_until_idle();  // ⛔ Block until complete

// Stage 3: Streamer
kpu->start_row_stream(...);
kpu->run_until_idle();  // ⛔ Block until complete
```

### After (Autonomous)
```cpp
// Program all stages upfront
orch.await({}, [&](){ /* DMA */ });
orch.await({DMA_DONE}, [&](){ /* BlockMover */ });
orch.await({BLOCK_DONE}, [&](){ /* Streamer */ });

// Single execution loop - all engines work concurrently
while (!orch.is_complete()) {
    kpu->step();
    orch.step();
}
```

## Benefits of Autonomous Model

### 1. True Concurrency
Multiple engines can be active simultaneously (e.g., DMA can transfer weights while BlockMover is processing input)

### 2. Realistic Timing
Cycle-accurate simulation reflects real hardware parallelism

### 3. No Manual Staging
No host read/write operations between pipeline stages

### 4. Dependency-Driven
Operations start automatically when dependencies are satisfied

### 5. Pipeline Efficiency
Later stages can start before earlier stages fully complete (when tiling/buffering allows)

### 6. Educational Value
Code clearly shows how real hardware orchestration works

## Future Extensions

### Multi-Tile Parallelism
```cpp
// Process different batches on different tiles concurrently
for (size_t tile = 0; tile < num_tiles; ++tile) {
    std::string prefix = "tile" + std::to_string(tile) + "_";
    orch.await({prefix + "data_ready"}, [&, tile](){
        kpu->start_matmul(tile, ...);
    });
}
```

### Double Buffering
```cpp
// While computing batch N, stream batch N+1
orch.await({COMPUTE_N_STARTED}, [&](){
    stream_batch_n_plus_1();
});
```

### Pipelined Execution
```cpp
// Layer 2 can start as soon as layer 1 produces first tile
orch.await({LAYER1_FIRST_TILE_DONE}, [&](){
    start_layer2_compute();
});
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Implement `AutonomousOrchestrator` class
2. Add signal emission to existing component callbacks
3. Test basic sequential pipeline

### Phase 2: MLP Implementation
1. Port `execute_mlp_layer()` to autonomous model
2. Verify correctness against original implementation
3. Measure cycle counts and compare

### Phase 3: Validation
1. Add timing analysis to compare GOD mode vs autonomous
2. Verify true concurrency through component activity traces
3. Document performance characteristics

### Phase 4: Extensions
1. Add multi-layer MLP execution
2. Implement double-buffering patterns
3. Explore multi-tile parallelism

## Conclusion

The autonomous execution model represents a significant step toward accurate hardware simulation. By eliminating centralized orchestration and implementing realistic synchronization primitives, we create a model that:

- **Accurately reflects** how real hardware operates
- **Enables true concurrency** and parallel execution
- **Simplifies reasoning** about data flow dependencies
- **Provides educational value** for understanding hardware architectures
- **Extends naturally** to more complex scenarios

The recommended Option 2 (Event/Signal System) provides the best balance of accuracy, implementation complexity, and educational value.
