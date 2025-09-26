# StorageScheduler Tests - Autonomous storage scheduling and multi-bank buffer memory

# Basic storage scheduler functionality tests
kpu_add_component_test(storage_scheduler_basic_test "eddo"
    test_storage_scheduler_basic.cpp
)

# Storage Scheduler workflow tests
kpu_add_component_test(eddo_workflow_test "eddo"
    test_eddo_workflow.cpp
)

# Performance and benchmarking tests
kpu_add_component_test(storage_scheduler_performance_test "eddo"
    test_storage_scheduler_performance.cpp
)
