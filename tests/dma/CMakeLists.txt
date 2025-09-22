# DMA Tests

# Basic DMA functionality tests
kpu_add_component_test(dma_basic_test "dma"
    test_dma_basic.cpp
)

# DMA performance and scalability tests
kpu_add_component_test(dma_performance_test "dma"
    test_dma_performance.cpp
)

# DMA tensor movement and ML workload tests
kpu_add_component_test(dma_tensor_test "dma"
    test_dma_tensor_movement.cpp
)

# DMA debug tests
kpu_add_component_test(dma_debug_test "dma"
    test_dma_debug.cpp
)