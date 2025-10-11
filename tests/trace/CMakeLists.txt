# Trace system tests

set(TRACE_TEST_SOURCES
    test_dma_tracing.cpp
)

# Create test executable (note: different name from custom target)
add_executable(trace_tests ${TRACE_TEST_SOURCES})

target_link_libraries(trace_tests
    PRIVATE
        Catch2::Catch2WithMain
        kpu_trace
        kpu_datamovement_components
        kpu_memory_components
)

target_include_directories(trace_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

# Set folder organization
set_target_properties(trace_tests PROPERTIES FOLDER "Tests/trace")

# Add to CTest with trace label
add_test(NAME trace_tests COMMAND trace_tests)
set_tests_properties(trace_tests PROPERTIES LABELS "trace;dma")

# Install test executable (optional)
install(TARGETS trace_tests
    DESTINATION bin/tests
    OPTIONAL
)
