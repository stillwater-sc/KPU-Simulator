# DMA Tests

# Basic DMA functionality tests
kpu_add_component_test(dma_basic_test "dma"
    test_dma_basic.cpp
)

# DMA performance and scalability tests
kpu_add_component_test(dma_performance_test "dma"
    test_dma_performance.cpp
)

# Mark performance test with additional label and timeout
set_tests_properties(dma_performance_test PROPERTIES
    LABELS "dma;performance"
    TIMEOUT 300
)

# DMA tensor movement and ML workload tests
kpu_add_component_test(dma_tensor_test "dma"
    test_dma_tensor_movement.cpp
)

# Set timeout for tensor test
set_tests_properties(dma_tensor_test PROPERTIES
    TIMEOUT 120
)

# DMA debug tests
kpu_add_component_test(dma_debug_test "dma"
    test_dma_debug.cpp
)