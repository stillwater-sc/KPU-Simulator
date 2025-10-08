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

# Mark performance test with additional label to exclude from CI
set_tests_properties(storage_scheduler_performance_test PROPERTIES
    LABELS "eddo;performance"
    TIMEOUT 600
)
