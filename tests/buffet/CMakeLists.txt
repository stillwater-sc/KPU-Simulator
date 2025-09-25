# MemoryOrchestrator Tests - EDDO and multi-bank buffer memory

# Basic memory orchestrator functionality tests
kpu_add_component_test(memory_orchestrator_basic_test "memory_orchestrator"
    test_orchestrator_basic.cpp
)

# EDDO workflow tests
kpu_add_component_test(eddo_workflow_test "eddo"
    test_eddo_workflow.cpp
)

# Performance and benchmarking tests
kpu_add_component_test(memory_orchestrator_performance_test "benchmark"
    test_orchestrator_performance.cpp
)
