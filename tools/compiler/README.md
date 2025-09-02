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
