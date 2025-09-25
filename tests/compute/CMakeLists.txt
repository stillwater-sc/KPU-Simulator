# Compute Engine Tests

kpu_add_component_test(compute_basic_test "compute"
    test_compute_basic.cpp
)

# Systolic Array Tests

kpu_add_component_test(systolic_array_test "systolic_array"
    test_systolic_array.cpp
)

kpu_add_component_test(matmul_tau111_s001_test "matmul_tau111_s001"
    test_matmul_tau111_s001.cpp
)
