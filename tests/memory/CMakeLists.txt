# Memory Tests

# Debug test (minimal, verbose output for diagnosing segfaults)
add_executable(test_memory_map_debug
    test_memory_map_debug.cpp
)

target_link_libraries(test_memory_map_debug
    PRIVATE
        kpu_memory_components
        Catch2::Catch2WithMain
)

target_include_directories(test_memory_map_debug
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(test_memory_map_debug PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    FOLDER "Tests/memory"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Memory mapping tests
add_executable(test_memory_map
    test_memory_map.cpp
)

target_link_libraries(test_memory_map
    PRIVATE
        kpu_memory_components
        Catch2::Catch2WithMain
)

target_include_directories(test_memory_map
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(test_memory_map PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    FOLDER "Tests/memory"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Sparse memory tests
add_executable(test_sparse_memory
    test_sparse_memory.cpp
)

target_link_libraries(test_sparse_memory
    PRIVATE
        kpu_memory_components
        Catch2::Catch2WithMain
)

target_include_directories(test_sparse_memory
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(test_sparse_memory PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    FOLDER "Tests/memory"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# External memory sparse tests
add_executable(test_external_memory_sparse
    test_external_memory_sparse.cpp
)

target_link_libraries(test_external_memory_sparse
    PRIVATE
        kpu_memory_components
        Catch2::Catch2WithMain
)

target_include_directories(test_external_memory_sparse
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(test_external_memory_sparse PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    FOLDER "Tests/memory"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Add tests to CTest
add_test(NAME MemoryMapDebug COMMAND test_memory_map_debug)
add_test(NAME MemoryMap COMMAND test_memory_map)
add_test(NAME SparseMemory COMMAND test_sparse_memory)
add_test(NAME ExternalMemorySparse COMMAND test_external_memory_sparse)

# Set test labels for filtering
set_tests_properties(MemoryMapDebug PROPERTIES LABELS "memory;memmap;debug")
set_tests_properties(MemoryMap PROPERTIES LABELS "memory;memmap")
set_tests_properties(SparseMemory PROPERTIES LABELS "memory;sparse")
set_tests_properties(ExternalMemorySparse PROPERTIES LABELS "memory;external;datacenter")
