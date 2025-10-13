# DMA Test Suite Overview

## Test Organization

The DMA test suite is organized into functional categories covering both legacy and modern APIs:

## 1. Basic Functionality (test\_dma\_basic.cpp)



&nbsp; - Transfer Operations: External↔Scratchpad, different memory banks

&nbsp; - Queue Management: Multiple queued transfers, FIFO ordering

&nbsp; - Data Integrity: Various transfer sizes (1B to 64KB), pattern verification

&nbsp; - Error Handling: Invalid addresses, IDs, boundary conditions

&nbsp; - State Management: Busy/idle status, reset functionality

&nbsp; - Concurrent Operations: Multiple DMA engines working simultaneously

&nbsp; - Matrix Data Movement: Floating-point matrix transfers for ML workloads



&nbsp; 2. Performance Testing (test\_dma\_performance.cpp)



&nbsp; - Transfer Size Scaling: Performance analysis from 1KB to 1MB

&nbsp; - Concurrent Transfer Scaling: Multi-DMA efficiency testing

&nbsp; - Memory Bank Distribution: Comparing single vs distributed bank strategies

&nbsp; - Large Dataset Streaming: 8MB dataset in 256KB chunks (L3→L2 simulation)

&nbsp; - Benchmarking: Catch2 benchmark integration for precise measurements



&nbsp; 3. Tensor Movement (test\_dma\_tensor\_movement.cpp)



&nbsp; - Basic Matrix Transfer: Standard tensor movement patterns

&nbsp; - Multi-Matrix Batches: Batch processing simulation

&nbsp; - Tiled Matrix Transfer: 2D tiling for large tensors (hardware acceleration patterns)

&nbsp; - Convolution Data Movement: CNN layer data streaming simulation

&nbsp; - Pipeline Simulation: Multi-stage ML pipeline with overlapped data movement

&nbsp; - Memory Bank Optimization: Sequential vs interleaved data placement strategies



&nbsp; Key Test Features



&nbsp; Real-World ML Scenarios



&nbsp; - Convolution Layers: Input feature map streaming (224×224×64 channels)

&nbsp; - Matrix Multiplication: Large matrix tiling (256×256 → 64×64 tiles)

&nbsp; - Pipeline Processing: Multi-stage computation with data overlap

&nbsp; - Batch Processing: Multiple tensor transfers for inference batches



&nbsp; Performance Analysis



&nbsp; - Throughput Measurement: MB/s calculations with wall-clock timing

&nbsp; - Scalability Testing: Concurrent DMA engine utilization

&nbsp; - Memory Bandwidth: Bank distribution optimization

&nbsp; - Cache Hierarchy Simulation: L3→L2→L1 data movement patterns



&nbsp; Error Conditions \& Edge Cases



&nbsp; - Boundary Testing: Out-of-bounds addresses and sizes

&nbsp; - Resource Validation: Invalid DMA/memory/scratchpad IDs

&nbsp; - Data Integrity: Pattern verification across all transfer sizes

&nbsp; - State Consistency: Reset and error recovery testing



## 4. Address-Based API (test\_dma\_address\_based.cpp) **[NEW - Recommended]**

The modern address-based API following industry standards (Intel I/OAT, ARM PL330, AMD SDMA):

### Core Features

- **Pure Address-Based Transfers**: Use unified address space instead of (type, id, offset) tuples
- **AddressDecoder Integration**: Automatic routing based on memory map configuration
- **Hardware Topology Independence**: Applications don't need to know physical memory layout
- **Virtual Memory Support**: Enable dynamic memory remapping without breaking DMA programs

### Test Coverage

1. **Basic Address-Based Transfer**
   - Simple transfer using only addresses
   - Automatic memory type and ID resolution via AddressDecoder

2. **Hardware Topology Independence**
   - Same DMA program works with different memory configurations
   - Memory map abstraction shields applications from hardware details

3. **Memory Map Visualization**
   - `AddressDecoder::to_string()` for debugging memory layouts
   - Clear visualization of address ranges and their mappings

4. **Error Handling - Unmapped Addresses**
   - Graceful failure when addresses aren't in memory map
   - Descriptive error messages with available address ranges

5. **Error Handling - Cross-Region Transfers**
   - Prevent transfers that span multiple memory regions
   - Detect invalid transfer ranges early

6. **Error Handling - Decoder Not Configured**
   - Clear error when AddressDecoder isn't attached to DMA engine
   - Guides user to proper setup

7. **Multiple Transfers Across Hierarchy**
   - Bank → Scratchpad, Bank → L2, L3 → Scratchpad
   - Tests full memory hierarchy routing

8. **Side-by-Side API Comparison**
   - Direct comparison of type-based vs address-based APIs
   - Demonstrates simplicity and clarity benefits

9. **Virtual Memory Simulation**
   - Simulates dynamic memory allocation with address-based interface
   - Shows how VM remapping doesn't break DMA programs

10. **Dynamic Memory Allocation Pattern**
    - Demonstrates malloc-like memory allocation abstraction
    - Applications use logical addresses, hardware handles routing

### Migration Benefits Demonstrated

| Aspect | Type-Based API | Address-Based API |
|--------|----------------|-------------------|
| Parameters per transfer | 7 (type, id, addr, type, id, addr, size) | 3 (src_addr, dst_addr, size) |
| Hardware coupling | Tight (requires topology knowledge) | Loose (topology-agnostic) |
| Virtual memory support | No | Yes |
| Industry standard | No | Yes (Intel, ARM, AMD, Xilinx, Google) |
| Error messages | Cryptic ID/type errors | Clear address-based errors with context |
| Code readability | Complex, verbose | Simple, intuitive |

### Running Address-Based Tests

```bash
# Run only address-based API tests
ctest -R dma_address_based_test -V

# View test source for migration examples
cat tests/dma/test_dma_address_based.cpp
```

### See Also

- **Documentation**: `docs/dma-engine-architecture.md` - Full API documentation and migration guide
- **Industry Analysis**: `docs/dma-architecture-comparison.md` - Comparison with Intel, ARM, AMD, Xilinx, Google
- **Migration Examples**: All tests show both APIs side-by-side for easy conversion reference

