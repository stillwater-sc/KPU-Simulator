# Session Log: 2025-12-01 - Tile Layout Policies and Realistic Timing Model

## Overview

This session addressed critical issues in the concurrent execution model where resource utilization was incorrect, and added realistic clock domain and bandwidth modeling based on LPDDR5X memory specifications.

## Problem Statement

The concurrent execution display was showing non-sensical schedules:
- BM[2], BM[3], STR[2], STR[3] showed 0% utilization
- DMA operations appeared sequential when they should be parallel
- First 30 cycles showed only DMA0 active
- A and B tiles for the same iteration were landing on the same memory channel

## Root Cause Analysis

1. **Hash-based Channel Selection**: The original `select_dma_channel()` used a hash that caused A[ti,tk] and B[tk,tj] for the same iteration to land on the same channel:
   ```cpp
   hash = matrix * 1000 + ti * 100 + tj * 10 + tk
   // A[0,0] -> hash 0 % 4 = 0
   // B[0,0] -> hash 1000 % 4 = 0  // CONFLICT!
   ```

2. **Static Resource Assignment**: BlockMover and Streamer selection used `l3_tile_id` and `l2_bank_id` which were always 0 in the program builder.

## Solution: Tile Layout Abstraction

### Four Configurable Layout Policies

1. **MATRIX_PARTITIONED** (Default)
   - Dedicates channels to specific matrices
   - A on channels {0,1}, B on channels {2,3}
   - Simple, predictable, 0% conflicts
   - Trade-off: May underutilize some channels

2. **ROUND_ROBIN**
   - Distributes tiles sequentially across all channels
   - Even distribution across resources
   - ~25% conflict rate (A and B may collide)

3. **ITERATION_AWARE**
   - A tiles always on even channels (0, 2)
   - B tiles always on odd channels (1, 3)
   - Guaranteed 0% conflicts
   - Optimal for maximum bandwidth

4. **HARDWARE_INTERLEAVED**
   - Address bits determine channel: `(addr / granularity) % num_channels`
   - Matches real hardware interleaving behavior
   - ~5% fragmentation overhead
   - 0% conflicts when tile sizes are aligned

### Key Files Created/Modified

- `include/sw/kpu/isa/tile_layout.hpp` - Base class and all 4 implementations
- `src/isa/tile_layout.cpp` - Full implementation with conflict analysis
- `include/sw/kpu/isa/concurrent_executor.hpp` - Updated ResourceConfig
- `src/isa/concurrent_executor.cpp` - Integrated TileLayout
- `examples/basic/tile_layout_test.cpp` - Comparison tool
- `examples/basic/concurrent_execution_debug.cpp` - Debug tool
- `docs/MEMORY_INTERLEAVING_DESIGN.md` - Design document

## Realistic Clock Domain Model

Based on hardware analysis:

| Domain | Clock | Cycle Time | Bus Width | Bandwidth |
|--------|-------|------------|-----------|-----------|
| Compute (ALUs) | 2.0 GHz | 500 ps | - | - |
| L1/L2/STR/BM | 500 MHz | 2 ns | 64 bytes | 32 GB/s |
| L3/DMA | 250 MHz | 4 ns | 64 bytes | 16 GB/s |

### Bandwidth Justification

- **Systolic array demand**: 16×16 × 4 bytes × 2 GHz = 256 GB/s ingress
- **L2 banks**: 8 banks × 32 GB/s = 256 GB/s (matches systolic demand)
- **DMA**: 64-byte burst per cycle @ 250 MHz = 16 GB/s per channel
- **Total external**: 4 channels × 16 GB/s = 64 GB/s

## Results

### Before Fix
```
Execution: 19664 cycles
DMA utilization: 14.8%
BM[2], BM[3]: 0% utilization
STR[2], STR[3]: 0% utilization
```

### After Fix
```
Execution: 38912 cycles (with realistic bandwidth)
DMA utilization: 23.7%
All BM[0-3]: 10.5% utilization
All STR[0-3]: 11.8% utilization
```

Note: Cycle count increased due to reduced bandwidth (16 GB/s vs 50 GB/s), but now accurately reflects real hardware.

### Timeline Verification

```
Cycles 0-255:   DMA[0] (A) and DMA[2] (B) running in parallel
Cycles 256-383: BM[0] (A) and BM[2] (B) running in parallel
Cycles 384-511: STR[0] (A) and STR[2] (B) running in parallel
```

## Enhanced Display Output

The timeline now shows:
- Clock domain legend with frequencies and cycle times
- Total execution time in ns/µs
- Scale mapping cycles to real time
- Aggregate bandwidth per resource type

Example output:
```
Clock Domains:
  DMA/L3:     250 MHz (4.0 ns/cycle), 64-byte bus = 16.0 GB/s/channel
  BM/L2:      500 MHz (2.0 ns/cycle), 64-byte bus = 32.0 GB/s/mover
  STR/L1:     500 MHz (2.0 ns/cycle), 64-byte bus = 32.0 GB/s/streamer
  Compute:    2000 MHz (0.50 ns/cycle), 16x16 systolic array

Timeline: 38912 DMA cycles = 155648.0 ns (155.65 µs)
```

## Future Considerations

1. **Pipeline Granularity**: 256 cycles (1024 ns) per DMA transfer creates large bubbles downstream. Consider:
   - Smaller tile sizes
   - Double/triple buffering
   - Pipelined DMA with partial transfers

2. **Ring Bus Modeling**: The design assumes a ring bus for contention-free L3 access. This should be modeled for accurate timing.

3. **Compute Utilization**: Currently not tracked. Need to add systolic array occupancy to the timeline.

## Testing

```bash
# Test tile layout policies
./examples/basic/example_tile_layout_test

# Debug concurrent execution
./examples/basic/example_concurrent_execution_debug

# Full matmul example with timeline
./examples/basic/example_data_movement_isa_matmul
```

## Commits

Changes ready for commit in working directory.
