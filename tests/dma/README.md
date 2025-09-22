\#   Test Suite Overview



&nbsp; 1. Basic Functionality (test\_dma\_basic.cpp)



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

