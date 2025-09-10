# tests/memory/CMakeLists.txt
# Memory Tests

# Memory allocation and management tests
add_kpu_test(memory_allocation_test "memory"
    test_memory_allocation.cpp
    test_memory_pool.cpp
    test_memory_alignment.cpp
)

# Memory safety and bounds checking tests  
# add_kpu_test(memory_safety_test "memory"
#     test_memory_bounds.cpp
#     test_memory_leaks.cpp
# )

# Memory performance tests
# add_kpu_test(memory_performance_test "memory"
#     test_memory_bandwidth.cpp
#     test_memory_latency.cpp
# )