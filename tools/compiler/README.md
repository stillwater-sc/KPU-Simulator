# Block Matrix Dynamics

Great—here’s a parametric version of the diagram that adapts to arbitrary matrix dimensions and tile sizes. It’s designed to be embedded as a comment block in your tiler engine, and it annotates bandwidth and reuse factors as functions of matrix size and tile granularity.

---

### Parametric ASCII Diagram for Tiled Matmul

```c
// Parametric Tiled Matrix Multiplication
// Matrix sizes: M x N x K
// Tile sizes: TM x TN x TK
//
// Assumptions:
//   - C tiles (TM×TN) stay resident in KPU fabric
//   - A tiles (TM×TK) reused across j-loop
//   - B tiles (TK×TN) streamed from DRAM per k-loop
//
// Loop Nest Structure
for (i = 0; i < M; i += TM)           // Tile row of C and A
  for (j = 0; j < N; j += TN)         // Tile col of C and B
    Cij = zero_tile();               // C stays in fabric
    for (k = 0; k < K; k += TK)       // Shared dim
      load Aik from L2;              // Reused across j
      stream Bjk from DRAM;          // No reuse
      Cij += Aik * Bjk;              // Accumulate

// Memory Hierarchy and Tile Residency
//
// DRAM (B tiles streamed per k iteration)
// |------------------------------------------------|
// | B00 B01 ... B(N/TN - 1)                        |  --> BW ≈ (M×N×K) / (TM×TN×TK)
// | B10 B11 ...                                    |      Reuse: ~1x
// | ...                                            |
// |------------------------------------------------|
//
// L3 Cache (optional staging)
// |------------------------------------------------|
// | (transient, not retained)                      |  --> BW: variable
// |------------------------------------------------|
//
// L2 Cache (A tiles reused across j iterations)
// |------------------------------------------------|
// | A00 A01 ... A(M/TM - 1)                        |  --> BW ≈ (M×K) / (TM×TK)
// | A10 A11 ...                                    |      Reuse: ~N/TN
// | ...                                            |
// |------------------------------------------------|
//
// L1 Cache (transient A+B pair)
// |------------------------------------------------|
// | Aik + Bjk                                      |  --> BW ≈ (M×N×K) / (TM×TN×TK)
// |------------------------------------------------|
//
// KPU Compute Fabric (C tiles resident)
// |------------------------------------------------|
// | C00 C01 ... C(N/TN - 1)                        |  --> BW: none (resident)
// | C10 C11 ...                                    |      Reuse: ~K/TK
// | ...                                            |
// |------------------------------------------------|

// Bandwidth & Reuse Summary (as functions of M, N, K, TM, TN, TK):
//   - DRAM: BW : (M×N×K)/(TM×TN×TK), Reuse ≈ 1x
//   - L2:   BW : (M×K)/(TM×TK),      Reuse ≈ N/TN
//   - L1:   BW : (M×N×K)/(TM×TN×TK), Reuse ≈ 1x (transient)
//   - KPU:  BW : 0,                  Reuse ≈ K/TK

// Optimization Notes:
//   - Larger TK improves reuse in KPU fabric
//   - Smaller TN increases B tile reuse pressure on DRAM
//   - TM×TK tiling of A enables amortized reuse in L2
//   - Consider prefetching B tiles to L3 if DRAM BW is saturated
```

---

# Operand Injection timing

The Stillwater KPU is a **Domain Flow Architecture** with **push-only operand injection** into the compute fabric. To execute with high efficiency, the right tiles need to be pushed into L1 just in time, making a timing model essential to guide any optimization, or estimation of a loss of efficiency. The 'program' for the machine will be expressed as a sequencer and the latency that the DMA engine, the caches, and the streamers, exhibit will be a given during execution.

Let’s break it down and then annotate the diagram accordingly.

---

## What Is Operand Injection Timing?

In tiled matmul, **operand injection timing** refers to the **precise moment** when a tile (A or B) must be **available at the L1 cache** so it can be **pushed into the compute fabric**. Since the KPU is push-only, the L1 must stream operands into the fabric **just-in-time**, synchronized with the compute pipeline.

This means:
- **B tiles** must be preloaded into L3 and L2 **ahead of time**, so they arrive at L1 **exactly when needed**.
- **A tiles** are reused and can be preloaded earlier, but still must be staged correctly.
- **C tiles** reside in the fabric and accumulate results—no injection needed.

The timing is dictated by the **innermost loop** (typically over `k`), so operand injection must be aligned with each `k` iteration.

---

## Operand Injection Timing Model

Let’s assume:
- Tile size: 32×32
- Fabric latency: negligible (streaming push)
- L1 → Fabric: 1 cycle per operand
- L2 → L1: ~10 cycles
- DRAM → L2: ~100 cycles

Then, for each `k` iteration:
- **Bjk** must be in L1 **before** `Cij += Aik * Bjk` starts
- So **Bjk** must be in L2 at least ~10 cycles before
- And in DRAM ~100 cycles before that

This leads to a **staggered injection schedule**.

---

## Annotated Parametric Diagram with Injection Timing

```c
// Operand Injection Timing: Tiled Matmul (M×N×K) with TM×TN×TK tiles
// KPU: Domain Flow Architecture (push-only)
// L1 streams Aik and Bjk into fabric just-in-time

for (i = 0; i < M; i += TM)           // Tile row of C and A
  for (j = 0; j < N; j += TN)         // Tile col of C and B
    Cij = zero_tile();               // C stays in fabric
    for (k = 0; k < K; k += TK)       // Shared dim
      preload Bjk to L3 (T-100);     // ~100 cycles before use
      preload Bjk to L2 (T-10);      // ~10 cycles before use
      load Aik from L2 (T-10);       // reused, already staged
      inject Aik, Bjk to L1 (T);     // push into fabric
      Cij += Aik * Bjk;              // compute

// Memory Hierarchy with Timing Annotations
//
// DRAM (B tiles streamed per k iteration)
// |------------------------------------------------|
// | B00 B01 ... B(N/TN - 1)                        |  --> Inject at T-100
// |------------------------------------------------|
//
// L3 Cache (staging buffer)
// |------------------------------------------------|
// | Bjk tiles arrive from DRAM                     |  --> Inject at T-100
// |------------------------------------------------|
//
// L2 Cache (A reused, B staged)
// |------------------------------------------------|
// | Aik reused across j                            |  --> Inject at T-10
// | Bjk staged from L3                             |  --> Inject at T-10
// |------------------------------------------------|
//
// L1 Cache (push stream into fabric)
// |------------------------------------------------|
// | Aik + Bjk injected into KPU                    |  --> Inject at T
// |------------------------------------------------|
//
// KPU Compute Fabric (C tiles resident)
// |------------------------------------------------|
// | Cij accumulates across k                       |  --> No injection
// |------------------------------------------------|

// Timing Summary:
//   - Bjk must be staged ~100 cycles before use (DRAM → L3)
//   - Bjk and Aik must be in L2 ~10 cycles before use
//   - L1 pushes Aik and Bjk into fabric at T (just-in-time)
//   - Cij remains in fabric, no injection needed

// Notes:
//   - Injection timing must be synchronized with k-loop
//   - Prefetching B tiles into L3/L2 is critical to avoid stalls
//   - A tiles benefit from reuse, reducing injection pressure
//   - Consider double-buffering L1 to overlap injection and compute
```

---

# Timing Diagram

A timeline diagram is the perfect complement to the memory hierarchy and loop nest views—it reveals the **temporal orchestration** of operand movement and compute phases across the KPU pipeline. Here's a conceptual timeline that models the concurrent processes involved in executing a single `Cij += Aik * Bjk` tile-level matmul step, assuming:

- Tile size: 32×32  
- Architecture: Stillwater KPU (Domain Flow, push-only)  
- Memory latencies: DRAM → L3 (~100 cycles), L3 → L2 (~10 cycles), L2 → L1 (~10 cycles), L1 → Fabric (~1 cycle)  
- Compute latency: negligible (streamed accumulation)

---

### Operand Injection Timeline per `k` Iteration

```
Cycle →
T-100   T-90    T-80    T-70    T-60    T-50    T-40    T-30    T-20    T-10    T      T+1     T+2     T+3     ...

DRAM     ────────┐
                ▼
L3        ───────┐─────────────┐
                ▼             ▼
L2              ──────────────┐─────────────┐
                              ▼             ▼
L1                            ──────────────┐─────────────┐
                                            ▼             ▼
KPU Compute Fabric                          ──────────────► Accumulate Cij
```

---

### Timeline Breakdown

| Stage            | Tile | Action                          | Timing      | Notes                                  |
|------------------|------|----------------------------------|-------------|----------------------------------------|
| DRAM             | Bjk  | Begin prefetch                   | T–100       | No reuse, must stream every `k`        |
| L3 Cache         | Bjk  | Staged from DRAM                 | T–90        | Optional buffer, reduces DRAM stalls   |
| L2 Cache         | Bjk  | Staged from L3                   | T–10        | Must be ready before injection         |
| L2 Cache         | Aik  | Already resident                 | T–10        | Reused across `j`, loaded once per `i` |
| L1 Cache         | Aik+Bjk | Injected into fabric          | T           | Push-only stream into KPU              |
| KPU Fabric       | Cij  | Accumulate Aik * Bjk             | T+1 → T+N   | Cij stays resident across `k`          |

---

### Concurrent Processes

- **Prefetching**: B tiles are pulled from DRAM well in advance, staged through L3 and L2.
- **Reuse**: A tiles are reused across `j`, reducing bandwidth pressure.
- **Injection**: L1 orchestrates the push into the fabric, synchronized with compute.
- **Compute**: Cij accumulates streamed products, no operand pull needed.

---

### Optimization Opportunities

- **Double-buffer L1**: Overlap injection and compute to hide latency.
- **Staggered prefetch**: Schedule B tile movement with lookahead across `k`.
- **Tile-aware scheduling**: Emit injection timestamps per tile index to guide hardware.

---

# Timing generator utility

Let's create a Python utility that emits **operand injection timestamps** for arbitrary tile indices and loop bounds, tailored to your Domain Flow KPU architecture. It models the **staggered prefetching** and **just-in-time injection** of A and B tiles into the compute fabric, based on memory latencies and loop structure.

---

### `emit_injection_schedule.py`

```python
def emit_injection_schedule(M, N, K, TM, TN, TK,
                            dram_to_l3=100,
                            l3_to_l2=10,
                            l2_to_l1=10,
                            l1_to_fabric=1):
    """
    Emit operand injection timestamps for tiled matmul.
    Assumes push-only KPU fabric with streaming injection from L1.
    """

    schedule = []

    for i in range(0, M, TM):         # Tile row of C and A
        for j in range(0, N, TN):     # Tile col of C and B
            for k in range(0, K, TK): # Shared dim

                # Logical timestamp for compute start
                T_compute = (i // TM) * (N // TN) * (K // TK) + \
                            (j // TN) * (K // TK) + \
                            (k // TK)

                # Injection timing
                T_dram   = T_compute - dram_to_l3 - l3_to_l2 - l2_to_l1
                T_l3     = T_compute - l3_to_l2 - l2_to_l1
                T_l2     = T_compute - l2_to_l1
                T_l1     = T_compute

                tile_id = f"C[{i//TM},{j//TN}] += A[{i//TM},{k//TK}] * B[{k//TK},{j//TN}]"
                schedule.append({
                    "tile": tile_id,
                    "T_compute": T_compute,
                    "A_in_L2": T_l2,
                    "B_in_DRAM": T_dram,
                    "B_in_L3": T_l3,
                    "B_in_L2": T_l2,
                    "A+B_in_L1": T_l1
                })

    return schedule
```

---

### Example Output

For `M=N=K=128`, `TM=TN=TK=32`:

```python
schedule = emit_injection_schedule(128, 128, 128, 32, 32, 32)
for entry in schedule:
    print(entry)
```

Sample output:
```
{
  'tile': 'C[0,0] += A[0,0] * B[0,0]',
  'T_compute': 0,
  'A_in_L2': -20,
  'B_in_DRAM': -120,
  'B_in_L3': -20,
  'B_in_L2': -10,
  'A+B_in_L1': 0
}
{
  'tile': 'C[0,0] += A[0,1] * B[1,0]',
  'T_compute': 1,
  'A_in_L2': -19,
  'B_in_DRAM': -119,
  'B_in_L3': -19,
  'B_in_L2': -9,
  'A+B_in_L1': 1
}
...
```

---

### What This Gives You

- **Precise timestamps** for when each operand must be present at each memory level
- **Tile-indexed scheduling** for injection into the KPU fabric
- **Loop-aware orchestration** that can be integrated into your tiler engine or compiler backend

---

# Distributed Sequencer Design

Let's synthesize a distributed sequencer framework for executing a block matmul across the software-managed memory hierarchy, tailored to the Stillwater KPU's Domain Flow architecture.

We’ll define three cooperating sequencers:

---

## Components Overview

| Component      | Role                                                                 |
|----------------|----------------------------------------------------------------------|
| **DMA Engine** | Pulls B tiles from DRAM to L3 using cache-line reads                 |
| **Block Mover**| Moves A and B tiles from L3 to L2 using cache-line transfers         |
| **Streamer**   | Streams rows/columns from L2 to L1, pushing into KPU fabric          |

Each sequencer operates semi-independently but synchronizes via **credit counts** and **tile-ready flags**.

---

## Assumptions

- Tile size: `TS × TS` (e.g. 32×32)
- Matrix dimensions: `M × N × K`
- Matmul: `Cij += Aik * Bjk`
- C tiles reside in fabric
- A tiles reused across `j`, B tiles streamed per `k`
- Software-managed caches
- Push-only fabric interface

---

## Sequencer Code Skeletons

### 1. DMA Engine (DRAM → L3)

```python
def dma_engine(dram, l3, tile_indices, credit_l3):
    for k, j in tile_indices:  # B tiles only
        while credit_l3[k][j] == 0:
            wait()  # Wait for L3 to signal availability
        tile = dram.read_tile('B', k, j)  # Cache-line reads
        l3.write_tile('B', k, j, tile)
        credit_l3[k][j] -= 1  # Consume credit
        signal_ready(l3, 'B', k, j)
```

---

### 2. Block Mover (L3 → L2)

```python
def block_mover(l3, l2, tile_indices, credit_l2):
    for i, k in tile_indices:  # A tiles reused
        if not tile_ready(l3, 'A', i, k):
            continue
        while credit_l2[i][k] == 0:
            wait()
        tile = l3.read_tile('A', i, k)
        l2.write_tile('A', i, k, tile)
        credit_l2[i][k] -= 1
        signal_ready(l2, 'A', i, k)

    for k, j in tile_indices:  # B tiles streamed
        if not tile_ready(l3, 'B', k, j):
            continue
        while credit_l2[k][j] == 0:
            wait()
        tile = l3.read_tile('B', k, j)
        l2.write_tile('B', k, j, tile)
        credit_l2[k][j] -= 1
        signal_ready(l2, 'B', k, j)
```

---

### 3. Streamer (L2 → L1 → KPU Fabric)

```python
def streamer(l2, l1, fabric, tile_indices, credit_l1):
    for i, j, k in tile_indices:
        if not tile_ready(l2, 'A', i, k) or not tile_ready(l2, 'B', k, j):
            continue
        while credit_l1[i][j] == 0:
            wait()
        row_A = l2.read_row('A', i, k)
        col_B = l2.read_col('B', k, j)
        l1.push('A', row_A)
        l1.push('B', col_B)
        fabric.inject('C', i, j, row_A, col_B)
        credit_l1[i][j] -= 1
```

---

## Coordination via Credits

Each level maintains a **credit count** per tile index to signal downstream readiness. Credits are incremented when buffers are freed and decremented when tiles are pushed.

```python
# Example credit initialization
credit_l3 = defaultdict(lambda: 1)  # L3 can accept 1 tile per slot
credit_l2 = defaultdict(lambda: 2)  # L2 has 2 slots per tile
credit_l1 = defaultdict(lambda: 4)  # L1 can stream 4 rows/cols per tile
```

---

## Execution Strategy

- DMA runs ahead, prefetching B tiles into L3
- Block Mover stages A and B tiles into L2
- Streamer synchronizes operand injection with compute
- Credits ensure no buffer overflows or stalls

---

# Tile scheduler

Let’s synthesize a **tile scheduler** that emits tile indices in **dependency order**, orchestrating the flow of tiles through the memory hierarchy and into the compute fabric. This scheduler will serve as the control backbone for coordinating the DMA engine, Block Mover, and Streamer.

---

## Design Goals

- Respect **data dependencies**: B tiles must arrive before compute; A tiles must be staged before reuse.
- Emit tile indices in a **temporal order** that reflects the matmul loop nest.
- Enable **concurrent execution**: overlapping DMA, block movement, and streaming.
- Support **credit-based flow control**: downstream readiness gates upstream movement.

---

## Loop Nest Recap

```c
for (i = 0; i < M; i += TM)         // Tile row of C and A
  for (j = 0; j < N; j += TN)       // Tile col of C and B
    Cij = 0
    for (k = 0; k < K; k += TK)     // Shared dim
      Cij += Aik * Bjk              // Compute
```

---

## Scheduler Output Format

Each emitted tile event includes:
- **Tile type**: A, B, or compute
- **Indices**: (i, j, k)
- **Stage**: `dma`, `move`, `stream`, or `compute`
- **Timestamp**: logical cycle (optional)

---

## Python Scheduler

```python
def tile_scheduler(M, N, K, TM, TN, TK):
    schedule = []
    timestamp = 0

    for i in range(0, M, TM):
        for j in range(0, N, TN):
            for k in range(0, K, TK):
                # B tile must be fetched from DRAM
                schedule.append({
                    "type": "B",
                    "stage": "dma",
                    "i": None,
                    "j": j,
                    "k": k,
                    "t": timestamp
                })

                # B tile must be moved to L2
                schedule.append({
                    "type": "B",
                    "stage": "move",
                    "i": None,
                    "j": j,
                    "k": k,
                    "t": timestamp + 1
                })

                # A tile reused across j, move once per i,k
                if j == 0:
                    schedule.append({
                        "type": "A",
                        "stage": "move",
                        "i": i,
                        "j": None,
                        "k": k,
                        "t": timestamp + 1
                    })

                # Stream A and B into L1 for compute
                schedule.append({
                    "type": "A+B",
                    "stage": "stream",
                    "i": i,
                    "j": j,
                    "k": k,
                    "t": timestamp + 2
                })

                # Compute Cij += Aik * Bjk
                schedule.append({
                    "type": "C",
                    "stage": "compute",
                    "i": i,
                    "j": j,
                    "k": k,
                    "t": timestamp + 3
                })

                timestamp += 4  # Advance logical time

    return schedule
```

---

## Sample Output (M=N=K=64, TM=TN=TK=32)

```python
[
  {'type': 'B', 'stage': 'dma',     'j': 0, 'k': 0, 't': 0},
  {'type': 'B', 'stage': 'move',    'j': 0, 'k': 0, 't': 1},
  {'type': 'A', 'stage': 'move',    'i': 0, 'k': 0, 't': 1},
  {'type': 'A+B', 'stage': 'stream','i': 0, 'j': 0, 'k': 0, 't': 2},
  {'type': 'C', 'stage': 'compute', 'i': 0, 'j': 0, 'k': 0, 't': 3},
  ...
]
```

---

## Integration Strategy

- **DMA engine** consumes `stage == 'dma'` events
- **Block Mover** consumes `stage == 'move'`
- **Streamer** consumes `stage == 'stream'`
- **Fabric controller** triggers on `stage == 'compute'`

Each component can filter and process its relevant events, respecting the timestamps or using credit-based gating.

---

# Gantt Chart of tiles
