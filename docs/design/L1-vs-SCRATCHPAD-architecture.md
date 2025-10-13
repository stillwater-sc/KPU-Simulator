# L1 vs SCRATCHPAD: Architectural Distinction

**Status**: Design Review
**Date**: 2025-01-13
**Issue**: Current implementation conflates two distinct memory spaces

---

## Executive Summary

The current KPU simulator incorrectly conflates **L1 streaming buffers** (compute datapath) with **Scratchpad page buffers** (memory controller). These are **two completely separate memory spaces** serving different purposes with different address spaces.

### The Bug

```cpp
// CURRENT (WRONG) - Conflates L1 and Scratchpad
enum class MemoryType {
    HOST_MEMORY,
    EXTERNAL,
    L3_TILE,
    L2_BANK,
    SCRATCHPAD        // ❌ Called "L1 scratchpad" - WRONG!
};
```

### The Fix

```cpp
// CORRECT - Separate L1 and Scratchpad
enum class MemoryType {
    HOST_MEMORY,      // Host DDR
    EXTERNAL,         // KPU GDDR6/HBM
    L3_TILE,          // L3 cache
    L2_BANK,          // L2 cache
    L1,               // L1 streaming buffers (compute fabric)
    SCRATCHPAD        // Page buffers (memory controller)
};
```

---

## Architecture Overview

### Two Independent Data Paths

```
┌────────────────────────────────────────────────────────────────┐
│                        KPU ARCHITECTURE                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         COMPUTE DATAPATH (Cache Hierarchy)              │  │
│  │                                                          │  │
│  │  External → L3 Tiles → L2 Banks → L1 Buffers → Compute │  │
│  │  Memory     (DMA)      (Block     (Streamers)   Fabric  │  │
│  │                        Movers)                           │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │      MEMORY CONTROLLER PATH (Page Buffering)            │  │
│  │                                                          │  │
│  │  External ←→ Scratchpad Page Buffers ←→ Host/DMA       │  │
│  │  Memory      (RMW page coherency)                       │  │
│  │                                                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 1. L1 Streaming Buffers (Compute Fabric)

### Purpose
High-bandwidth streaming buffers that feed data directly into systolic arrays and compute units.

### Physical Location
- **Integrated with compute fabric**
- Physically adjacent to systolic arrays
- Part of the compute tile

### Address Space
- **Linear byte addressing** (0x0000 - capacity)
- Small, fast buffers (typical: 32-64 KB per compute tile)
- Accessed by compute fabric and streamers

### Dataflow
```
L2 Cache → [Streamer] → L1 Buffer → Systolic Array → L1 Buffer → [Streamer] → L2 Cache
```

### Characteristics
- **Read latency**: 1-2 cycles
- **Write latency**: 1-2 cycles
- **Bandwidth**: Very high (matches compute fabric requirements)
- **Access pattern**: Streaming (row/column streaming for matrix operations)
- **Purpose**: Feed compute units with data

### Components That Use L1
- ✅ **Streamers**: L2 ↔ L1 data movement
- ✅ **Compute Fabric**: Direct read/write access
- ✅ **Systolic Arrays**: Read operands, write results

### Configuration Example
```cpp
struct KPUL1Config {
    std::string id;
    uint32_t capacity_kb{32};  // Small, fast buffers
};

// Typical configuration: 4 compute tiles, each with 32KB L1
for (int i = 0; i < 4; ++i) {
    KPUL1Config l1;
    l1.id = "l1_" + std::to_string(i);
    l1.capacity_kb = 32;
    kpu.memory.l1_buffers.push_back(l1);
}
```

---

## 2. Scratchpad Page Buffers (Memory Controller)

### Purpose
**Page buffering** to improve external memory efficiency by collating small reads/writes into page-coherent transactions.

### Physical Location
- **Integrated with memory controller**
- Part of the memory controller's internal state
- **NOT between L1 and compute fabric**

### Address Space
**THIS IS THE KEY DIFFERENCE:**

```cpp
// Scratchpad uses PAGE-ID addressing, not byte addressing!
struct ScratchpadAddress {
    uint32_t page_id;        // Which page buffer (NOT byte address)
    uint32_t offset_in_page; // Offset within the page
};
```

- **Page-ID based addressing** (maps to external memory pages)
- Typical page size: 4KB, 8KB, or 16KB
- Used for **read-modify-write (RMW)** operations

### Dataflow
```
External Memory ←→ Scratchpad Page Buffers ←→ DMA/Host
         ↑                    ↑
         │                    │
    Page-level           Small read/write
    transactions         operations collated
```

### Characteristics
- **Purpose**: Aggregate small operations into page-sized transactions
- **Address space**: Page-IDs (not linear byte addresses)
- **Operation**: Read-Modify-Write (RMW) page coherency
- **Benefit**: Reduce external memory traffic, improve efficiency
- **Typical size**: 64KB - 256KB (holds multiple pages)

### Use Cases

#### Example 1: Small Scattered Writes
```
Problem: Application needs to write 100 bytes each to 20 different locations
         Direct to external memory: 20 × 4KB page writes = 80KB traffic

Solution:
1. Scratchpad pulls affected pages
2. Accumulates writes in page buffers
3. Writes back complete pages
Result: Only modified pages written, much less traffic
```

#### Example 2: Row/Column Operations
```
Problem: Reading a column from row-major matrix requires many cache line reads

Solution:
1. Scratchpad buffers the relevant pages
2. Extracts column data from page buffers
3. Single page-sized transaction to external memory
```

### Components That Use Scratchpad
- ✅ **Memory Controller**: Direct page buffer management
- ✅ **DMA Engine**: For external memory page operations
- ❌ **NOT Streamers**: Streamers use L1, not Scratchpad
- ❌ **NOT Compute Fabric**: Compute reads from L1, not Scratchpad

### Configuration Example
```cpp
struct KPUScratchpadConfig {
    std::string id;
    uint32_t capacity_kb{64};   // Page buffer capacity
    uint32_t page_size_kb{4};   // Page size (typically 4KB)
};

// Typical configuration: 4 scratchpads for memory controller
for (int i = 0; i < 4; ++i) {
    KPUScratchpadConfig scratch;
    scratch.id = "scratch_" + std::to_string(i);
    scratch.capacity_kb = 64;    // Can hold 16 pages of 4KB
    scratch.page_size_kb = 4;
    kpu.memory.scratchpads.push_back(scratch);
}
```

---

## 3. Address Space Organization

### Unified Address Space (for DMA/L1)

```
+---------------------------+ 0x0000'0000
| Host Memory (4GB)         |
+---------------------------+ 0x1'0000'0000
| External Memory Bank 0    |
| (2GB)                     |
+---------------------------+ 0x1'8000'0000
| External Memory Bank 1    |
| (2GB)                     |
+---------------------------+ 0x2'0000'0000
| L3 Tiles (4 × 256KB)      |
+---------------------------+ 0x2'0010'0000
| L2 Banks (8 × 128KB)      |
+---------------------------+ 0x2'0020'0000
| L1 Buffers (4 × 32KB)     | ← NEW: L1 in unified address space
+---------------------------+ 0x2'0022'0000
| (Reserved/Future)         |
+---------------------------+
```

### Scratchpad Address Space (SEPARATE)

**Scratchpads do NOT participate in unified address space!**

```cpp
// Scratchpad operations use page-ID, not global address
class MemoryController {
public:
    // Load external memory page into scratchpad buffer
    void load_page(uint32_t scratchpad_id, uint32_t page_id,
                   Address external_page_addr);

    // Write modified page back to external memory
    void flush_page(uint32_t scratchpad_id, uint32_t page_id,
                    Address external_page_addr);

    // Read/write within page buffer
    void read_from_page(uint32_t scratchpad_id, uint32_t page_id,
                       uint32_t offset, void* data, size_t size);
    void write_to_page(uint32_t scratchpad_id, uint32_t page_id,
                      uint32_t offset, const void* data, size_t size);
};
```

---

## 4. Component Updates Required

### 4.1 Address Decoder

```cpp
enum class MemoryType {
    HOST_MEMORY,      // Host DDR (CPU-side)
    EXTERNAL,         // KPU external memory banks (GDDR6/HBM)
    L3_TILE,          // L3 cache tiles
    L2_BANK,          // L2 cache banks
    L1,               // ✅ NEW: L1 streaming buffers (compute fabric)
    // SCRATCHPAD is NOT here - uses page-ID addressing
};
```

**Note**: Scratchpad is intentionally NOT in `MemoryType` because it doesn't use the unified address space.

### 4.2 KPU Simulator Config

```cpp
struct Config {
    // ... existing fields ...

    // L1 streaming buffers (compute fabric)
    Size l1_buffer_count;        // ✅ NEW
    Size l1_buffer_capacity_kb;  // ✅ NEW
    Address l1_buffer_base;      // ✅ NEW (for unified address space)

    // Scratchpad page buffers (memory controller)
    Size scratchpad_count;       // KEEP (but clarify purpose)
    Size scratchpad_capacity_kb; // KEEP
    Size scratchpad_page_size_kb; // ✅ NEW (page size)
    // NO scratchpad_base - doesn't use unified address space!
};
```

### 4.3 Component Storage

```cpp
class KPUSimulator {
private:
    // Compute datapath (cache hierarchy)
    std::vector<ExternalMemory> memory_banks;
    std::vector<L3Tile> l3_tiles;
    std::vector<L2Bank> l2_banks;
    std::vector<L1Buffer> l1_buffers;     // ✅ NEW
    std::vector<ComputeFabric> compute_tiles;

    // Memory controller components
    std::vector<Scratchpad> scratchpads;  // KEEP (but fix semantics)
    std::vector<DMAEngine> dma_engines;

    // Data movement
    std::vector<BlockMover> block_movers; // L3 → L2
    std::vector<Streamer> streamers;      // L2 → L1
};
```

### 4.4 Streamer Component

**CRITICAL FIX**: Streamers target L1, NOT Scratchpad

```cpp
struct StreamConfig {
    // Source and destination
    size_t l2_bank_id;
    size_t l1_buffer_id;     // ✅ CHANGED from l1_scratchpad_id

    // Memory addresses
    Address l2_base_addr;
    Address l1_base_addr;

    // ... rest unchanged ...
};
```

### 4.5 Scratchpad Component (NEW Design)

```cpp
class Scratchpad {
private:
    size_t page_size_;                              // Page size (4KB, 8KB, etc.)
    size_t num_pages_;                              // Number of page buffers
    std::vector<std::vector<uint8_t>> page_buffers_; // Page buffer storage
    std::unordered_map<uint32_t, size_t> page_map_;  // page_id → buffer_index

public:
    Scratchpad(size_t capacity_kb, size_t page_size_kb);

    // Page operations (NOT byte-addressed)
    bool load_page(uint32_t page_id, const void* external_data);
    bool flush_page(uint32_t page_id, void* external_data);
    bool is_page_loaded(uint32_t page_id) const;

    // In-page operations
    void read_from_page(uint32_t page_id, uint32_t offset,
                       void* data, size_t size);
    void write_to_page(uint32_t page_id, uint32_t offset,
                      const void* data, size_t size);

    // Page buffer management
    void evict_page(uint32_t page_id);
    size_t get_page_count() const { return num_pages_; }
    size_t get_page_size() const { return page_size_; }
};
```

---

## 5. Usage Examples

### Example 1: Compute Datapath (Using L1)

```cpp
// Compute datapath: External → L3 → L2 → L1 → Compute

// 1. DMA: External → L3
Address ext_addr = kpu.get_external_bank_base(0) + 0x1000;
Address l3_addr = kpu.get_l3_tile_base(0) + 0x0;
kpu.start_dma_transfer(0, ext_addr, l3_addr, size);

// 2. BlockMover: L3 → L2
kpu.start_block_transfer(0, l3_tile_id, l3_offset,
                         l2_bank_id, l2_offset,
                         height, width, sizeof(float));

// 3. Streamer: L2 → L1 (NOT Scratchpad!)
Address l2_addr = kpu.get_l2_bank_base(0) + 0x0;
Address l1_addr = kpu.get_l1_buffer_base(0) + 0x0;  // ✅ NEW
kpu.start_row_stream(streamer_id, l2_bank_id, l1_buffer_id,
                     l2_addr, l1_addr, height, width, ...);

// 4. Compute: Read from L1
kpu.start_matmul(tile_id, l1_buffer_id, m, n, k,
                 a_addr, b_addr, c_addr);
```

### Example 2: Memory Controller Path (Using Scratchpad)

```cpp
// Memory controller: Small writes aggregated via page buffers

// 1. Application wants to write 100 bytes to many scattered locations
std::vector<ScatteredWrite> writes = {
    {addr: 0x1000, data: [...], size: 100},
    {addr: 0x5000, data: [...], size: 100},
    {addr: 0x9000, data: [...], size: 100},
    // ... 20 total scattered writes
};

// 2. Memory controller uses scratchpad to aggregate
MemoryController mc;
for (const auto& write : writes) {
    uint32_t page_id = mc.addr_to_page_id(write.addr);
    uint32_t offset = write.addr % page_size;

    // Load page if not present
    if (!scratchpad.is_page_loaded(page_id)) {
        mc.load_page(scratchpad_id, page_id,
                     external_memory_base + page_id * page_size);
    }

    // Write to page buffer
    scratchpad.write_to_page(page_id, offset,
                            write.data, write.size);
}

// 3. Flush modified pages back to external memory
mc.flush_all_dirty_pages(scratchpad_id);
```

---

## 6. Memory Map Display (Updated)

```
Unified Address Space Memory Map:
  +---------------------------------------------------------+
  | Host Memory                                             |
  |   Region[0]:  0x0000000000  (4096 MB)                   |
  +---------------------------------------------------------+
  | External Memory (GDDR6)                                 |
  |   Bank[0]:    0x0100000000  (2048 MB)                   |
  |   Bank[1]:    0x0180000000  (2048 MB)                   |
  +---------------------------------------------------------+
  | L3 Cache Tiles                                          |
  |   Tile[0]:    0x0200000000  (256 KB)                    |
  |   Tile[1]:    0x0200040000  (256 KB)                    |
  |   Tile[2]:    0x0200080000  (256 KB)                    |
  |   Tile[3]:    0x02000c0000  (256 KB)                    |
  +---------------------------------------------------------+
  | L2 Cache Banks                                          |
  |   Bank[0]:    0x0200100000  (128 KB)                    |
  |   Bank[1]:    0x0200120000  (128 KB)                    |
  |   ... (8 total)                                         |
  +---------------------------------------------------------+
  | L1 Streaming Buffers (Compute Fabric)                   | ← NEW
  |   L1[0]:      0x0200200000  (32 KB)                     |
  |   L1[1]:      0x0200208000  (32 KB)                     |
  |   L1[2]:      0x0200210000  (32 KB)                     |
  |   L1[3]:      0x0200218000  (32 KB)                     |
  +---------------------------------------------------------+

Scratchpad Page Buffers (Memory Controller):
  +---------------------------------------------------------+
  | NOT in unified address space - uses page-ID addressing  |
  +---------------------------------------------------------+
  | Scratchpad[0]: 64 KB (16 pages × 4KB)                   |
  | Scratchpad[1]: 64 KB (16 pages × 4KB)                   |
  | Scratchpad[2]: 64 KB (16 pages × 4KB)                   |
  | Scratchpad[3]: 64 KB (16 pages × 4KB)                   |
  +---------------------------------------------------------+
```

---

## 7. Implementation Checklist

### Phase 1: L1 Infrastructure
- [ ] Create `L1Buffer` component class (similar to L2Bank)
- [ ] Add `l1_buffer_count`, `l1_buffer_capacity_kb`, `l1_buffer_base` to `KPUSimulator::Config`
- [ ] Add `std::vector<L1Buffer> l1_buffers` to `KPUSimulator`
- [ ] Update address space initialization to include L1 region
- [ ] Add `get_l1_buffer_base()` address helper
- [ ] Add L1 to unified address decoder

### Phase 2: Scratchpad Redesign
- [ ] Redesign `Scratchpad` class for page-ID addressing
- [ ] Add `page_size_kb` to `KPUScratchpadConfig`
- [ ] Implement page buffer management (load/flush/evict)
- [ ] Remove scratchpad from unified address space
- [ ] Create `MemoryController` class to manage scratchpad operations

### Phase 3: Streamer Updates
- [ ] Update `StreamConfig` to use `l1_buffer_id` instead of `l1_scratchpad_id`
- [ ] Update all streamer operations to target L1 buffers
- [ ] Update `KPUSimulator::start_row_stream()` and `start_column_stream()` signatures
- [ ] Update documentation and comments

### Phase 4: Integration
- [ ] Update `host_t100_autonomous.cpp` to use L1 correctly
- [ ] Update all tests to distinguish L1 from scratchpad
- [ ] Update memory map display
- [ ] Update Python bindings
- [ ] Update documentation

### Phase 5: Validation
- [ ] Test compute datapath (External → L3 → L2 → L1 → Compute)
- [ ] Test memory controller path with page operations
- [ ] Verify address space correctness
- [ ] Performance validation

---

## 8. Open Questions

1. **Scratchpad Page Size**: Should this be configurable? (4KB, 8KB, 16KB?)
2. **Page Replacement Policy**: LRU, FIFO, or explicit management?
3. **Coherency**: Do we need coherency between L1 and Scratchpad? (Answer: NO - they're completely separate)
4. **DMA Access**: Can DMA directly access L1 buffers for results? Or only via L2?
5. **Multi-scratchpad**: Do different scratchpads handle different page ranges?

---

## 9. Benefits of This Design

### Correctness
✅ Accurately models real hardware architecture
✅ Separates compute datapath from memory controller operations
✅ Proper address space semantics

### Performance
✅ L1 optimized for streaming to compute fabric
✅ Scratchpad optimized for page-level RMW operations
✅ Reduced external memory traffic via page aggregation

### Flexibility
✅ Independent sizing of L1 vs Scratchpad
✅ Configurable page sizes for different workloads
✅ Clear separation of concerns

---

## 10. Summary

| Aspect | L1 Streaming Buffers | Scratchpad Page Buffers |
|--------|---------------------|------------------------|
| **Purpose** | Feed compute fabric | Aggregate memory operations |
| **Location** | Compute tile | Memory controller |
| **Address Space** | Unified (byte addressing) | Page-ID (not in unified space) |
| **Size** | Small (32-64 KB) | Medium (64-256 KB) |
| **Access Pattern** | Streaming | Page-level RMW |
| **Used By** | Streamers, Compute | DMA, Memory Controller |
| **Datapath** | L2 → L1 → Compute | External ↔ Scratchpad ↔ Host |

**Key Insight**: These are **completely different memory spaces** that happen to both be on-chip. They serve different masters (compute vs memory controller) and have different address semantics (byte vs page-ID).
