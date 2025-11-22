# tests/compiler/CMakeLists.txt

# Graph loader tests
add_executable(graph_loader_test
    test_graph_loader.cpp
)

target_link_libraries(graph_loader_test
    PRIVATE
        kpu_compiler
        Catch2::Catch2WithMain
)

target_include_directories(graph_loader_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Copy test graphs to build directory for testing
add_custom_command(TARGET graph_loader_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/test_graphs
        ${CMAKE_CURRENT_BINARY_DIR}/test_graphs
    COMMENT "Copying test_graphs to build directory"
)

# Register with CTest
# Set working directory to project root so test can find test_graphs/
add_test(NAME graph_loader_test
    COMMAND graph_loader_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

set_target_properties(graph_loader_test PROPERTIES
    FOLDER "Tests/Compiler"
)
