# StorageScheduler Tests - Autonomous storage scheduling and multi-bank buffer memory

# Basic storage scheduler functionality tests
kpu_add_component_test(storage_scheduler_basic_test "storage_scheduler"
    test_orchestrator_basic.cpp
)

# Storage workflow tests
kpu_add_component_test(storage_workflow_test "storage_workflow"
    test_eddo_workflow.cpp
)

# Performance and benchmarking tests
kpu_add_component_test(storage_scheduler_performance_test "benchmark"
    test_orchestrator_performance.cpp
)
