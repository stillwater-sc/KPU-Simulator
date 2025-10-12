# Compute Engine Tests

kpu_add_component_test(compute_basic_test "compute"
    test_compute_basic.cpp
)

# Systolic Array Tests

kpu_add_component_test(systolic_array_test "fabric"
    test_systolic_array.cpp
)

kpu_add_component_test(matmul_tau111_s001_test "fabric"
    test_matmul_tau111_s001.cpp
)

# Compute Fabric Tracing Tests

kpu_add_component_test(compute_fabric_tracing_test "fabric"
    test_compute_fabric_tracing.cpp
)
