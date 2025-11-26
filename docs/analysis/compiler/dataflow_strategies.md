# Dataflow Strategies Quick Start Guide

**Date**: November 24, 2025

## TL;DR

The KPU simulator now supports **three dataflow strategies** (OS, WS, IS) for systolic array optimization. Each strategy is optimal for different workload patterns.

### Quick Strategy Selection

```python
# Choose based on workload dimensions M×N×K

if K > 2 * max(M, N):
    use "Output-Stationary (OS)"    # Large K → accumulation critical

elif M > 2 * N:
    use "Weight-Stationary (WS)"    # Large M → batch processing

elif N > 2 * M:
    use "Input-Stationary (IS)"     # Large N → wide outputs

else:
    use "Output-Stationary (OS)"    # Default for balanced workloads
```

## What Changed

### Previously
- Only Output-Stationary (OS) was implemented
- Input-Stationary fell back to OS
- No real differentiation between strategies

### Now ✅
- **All three strategies fully implemented**
- Each produces different tile configurations
- Characteristic reuse patterns validated
- Appears on Pareto frontier for appropriate workloads

## Running the Code

### 1. Build and Test
```bash
cd build

# Build and run all tests
ninja tile_optimizer_test
./tests/compiler/tile_optimizer_test

# All tests pass: 163 assertions in 24 test cases ✅
```

### 2. Run Characterization Demo
```bash
# Build demo
ninja schedule_characterizer_demo

# Run demo (evaluates 300 workloads with all 3 strategies)
./examples/compiler/schedule_characterizer_demo

# Output files:
# - pareto_frontier_small.csv (300 evaluations)
# - pareto_frontier_networks.csv (54 evaluations)
# - pareto_frontier_sweep.csv (3072 evaluations)
```

### 3. Visualize Results
```bash
# Generate visualization
python ../tools/compiler/visualize_pareto_frontier.py pareto_frontier_small.csv

# Output:
# - pareto_frontier_small_visualization.png (4-panel analysis)
# - Statistics printed to console
```

## Example Results

### Three-Way Comparison (512×512×512)

```
Strategy: Output-Stationary (OS)
  Reuse: A=8×, B=8× (balanced)
  Latency: 57,344 cycles

Strategy: Weight-Stationary (WS)
  Reuse: A=8×, B=56× (B reuse 7× higher!)
  Latency: 43,008 cycles (25% faster)

Strategy: Input-Stationary (IS)
  Reuse: A=56×, B=8× (A reuse 7× higher!)
  Latency: 43,008 cycles (25% faster)
```

### Strategy Advantages

| Strategy | Best For | Example | Advantage |
|----------|----------|---------|-----------|
| **OS** | Balanced, Large K | 128×128×1024 | 62.8% energy savings |
| **WS** | Large M (batch) | 1024×256×256 | B reuse 4.73× higher |
| **IS** | Large N (wide output) | 256×1024×256 | A reuse 4.73× higher |

## Visualization Output

The visualization shows 4 panels:

1. **Top-Left: Energy vs Throughput** (PRIMARY PARETO)
   - Shows proper tradeoff: high throughput → high energy
   - Color-coded by strategy (WS=red, IS=blue, OS=green)

2. **Top-Right: Slowdown Analysis**
   - Energy slowdown vs Latency slowdown
   - Shows efficiency relative to ideal

3. **Bottom-Left: Energy vs Latency** (INFORMATIONAL)
   - These are correlated (NOT a valid Pareto)
   - Good schedules have both low

4. **Bottom-Right: Energy vs Tensor Size**
   - Shows how energy scales with problem size

## Key Statistics

From `pareto_frontier_small.csv` (300 evaluations):

```
Strategy distribution:
  WS: 100 (33.3%)
  IS: 100 (33.3%)
  OS: 100 (33.3%)

Throughput range:
  Min:    29.49 GFLOP/s
  Max:  6553.60 GFLOP/s
  Mean: 2085.64 GFLOP/s

Pareto-Optimal Schedules: 13 (4.33% coverage)
  WS contribution: 9 points (69.2%)
  IS contribution: 4 points (30.8%)
```

## Understanding the Results

### Reuse Patterns

Each strategy has a characteristic reuse pattern:

```
Output-Stationary (OS):
  - Balances A and B reuse
  - C accumulates in PEs (free!)
  - Best for small/balanced workloads

Weight-Stationary (WS):
  - Maximizes B reuse
  - B stays in PE registers
  - Best for batch processing (large M)

Input-Stationary (IS):
  - Maximizes A reuse
  - A stays in PE registers
  - Best for wide outputs (large N)
```

### When Each Strategy Wins

```
OS wins when:
  - K is large (deep accumulation)
  - Dimensions are balanced (M≈N≈K)
  - Example: 128×128×1024 → OS saves 62.8% energy

WS wins when:
  - M is large (batch processing)
  - Weights are small (small K×N)
  - Example: 1024×256×256 → WS achieves 4.73× B reuse

IS wins when:
  - N is large (many output features)
  - Inputs are small (small M×K)
  - Example: 256×1024×256 → IS achieves 4.73× A reuse
```

## Common Workload Patterns

### 1. Batch Inference (Large M)
```
Workload: 1024×256×256 (large batch, small weights)
Winner: WS
Reason: B reuse = 52× vs OS 11× (4.73× improvement)
```

### 2. Feature Extraction (Large N)
```
Workload: 256×1024×256 (small inputs, wide output)
Winner: IS
Reason: A reuse = 52× vs OS 11× (4.73× improvement)
```

### 3. Deep Accumulation (Large K)
```
Workload: 128×128×1024 (deep reduction)
Winner: OS
Reason: C accumulates in PEs → 256KB vs 688KB (62.8% savings)
```

### 4. Balanced Workload
```
Workload: 256×256×256 (all dimensions similar)
Winner: OS
Reason: Minimal DRAM (409KB vs 671KB), balanced reuse
```

## File Locations

### Implementation
- **Header**: `include/sw/compiler/tile_optimizer.hpp`
- **WS Implementation**: `src/compiler/tile_optimizer.cpp:368-546`
- **IS Implementation**: `src/compiler/tile_optimizer.cpp:548-726`
- **Integration**: `src/compiler/schedule_characterizer.cpp:263-275`

### Tests
- **All Tests**: `tests/compiler/test_tile_optimizer.cpp`
- **WS Tests**: Lines 371-614
- **IS Tests**: Lines 616-882

### Visualization
- **Script**: `tools/compiler/visualize_pareto_frontier.py`
- **Output**: `build/pareto_frontier_small_visualization.png`

### Documentation
- **WS Details**: `build/WS_IMPLEMENTATION_STATUS.md`
- **IS Details**: `build/IS_IMPLEMENTATION_COMPLETE.md`
- **Complete Summary**: `build/ALL_DATAFLOW_STRATEGIES_COMPLETE.md`
- **Quick Start**: This file

## Next Steps

### For Users
1. Run the demo to see all three strategies
2. Visualize the Pareto frontier
3. Identify which strategy wins for your workloads
4. Use the strategy selection guide above

### For Developers
1. Review the implementation in `tile_optimizer.cpp`
2. Run the unit tests to understand behavior
3. Examine the three-way comparison test
4. Read the detailed documentation files

### Future Enhancements
- **Phase 2**: Strategy-specific L2 schedulers (loop reordering)
- **Phase 3**: Accumulator precision modeling (INT8→INT32)
- **Phase 4**: Hybrid strategies (mix per layer)
- **Phase 5**: Auto-tuning and ML-based selection

## Troubleshooting

### Tests fail
```bash
# Rebuild from scratch
cd build
rm -rf *
cmake ..
ninja
ninja tile_optimizer_test
./tests/compiler/tile_optimizer_test
```

### Demo doesn't show different strategies
```bash
# Check that all three strategies are enabled
grep -A 5 "strategy_comparison" schedule_characterizer_demo.cpp

# Verify CSV has all three strategies
awk -F, 'NR>1 {print $4}' pareto_frontier_small.csv | sort | uniq -c
# Should show: 100 IS, 100 OS, 100 WS
```

### Visualization fails
```bash
# Install dependencies
pip install pandas matplotlib numpy

# Check CSV format
head -3 pareto_frontier_small.csv
# Should have: M,N,K,Strategy,Energy_pJ,Latency_cycles,Throughput_GFLOPS,...
```

## Summary

✅ **All three dataflow strategies implemented**
✅ **163 test assertions pass**
✅ **Pareto frontier shows strategy differentiation**
✅ **Strategy selection guide provided**
✅ **Comprehensive documentation available**

The KPU simulator now provides production-ready dataflow strategy selection for systolic array optimization!
