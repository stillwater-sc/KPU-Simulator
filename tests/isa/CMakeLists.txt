# ============================================================================
# tests/isa/CMakeLists.txt
# Tests for Data Movement ISA

# Test for Data Movement ISA and program execution
add_executable(test_data_movement_isa
    test_data_movement_isa.cpp
)

target_include_directories(test_data_movement_isa PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_data_movement_isa
    PRIVATE
    kpu_isa
    kpu_components
    kpu_trace
)

target_compile_features(test_data_movement_isa PUBLIC cxx_std_20)

# Add to CTest
add_test(NAME test_data_movement_isa COMMAND test_data_movement_isa)
set_tests_properties(test_data_movement_isa PROPERTIES LABELS isa)

# Set folder organization
set_target_properties(test_data_movement_isa PROPERTIES FOLDER "Tests/ISA")
