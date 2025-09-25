# Buffet Tests - EDDO and multi-bank buffer memory

# Basic buffet functionality tests
kpu_add_component_test(buffet_basic_test "buffet"
    test_buffet_basic.cpp
)

# EDDO workflow tests
kpu_add_component_test(eddo_workflow_test "eddo"
    test_eddo_workflow.cpp
)

# Performance and benchmarking tests
kpu_add_component_test(buffet_performance_test "benchmark"
    test_buffet_performance.cpp
)