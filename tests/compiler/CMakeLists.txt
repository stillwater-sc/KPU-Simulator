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

# Tile optimizer tests
add_executable(tile_optimizer_test
    test_tile_optimizer.cpp
)

target_link_libraries(tile_optimizer_test
    PRIVATE
        kpu_compiler
        Catch2::Catch2WithMain
)

target_include_directories(tile_optimizer_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Register with CTest
add_test(NAME tile_optimizer_test
    COMMAND tile_optimizer_test
)

set_target_properties(tile_optimizer_test PROPERTIES
    FOLDER "Tests/Compiler"
    LABELS "unit;compiler"
)

# Schedule generator tests
add_executable(schedule_generator_test
    test_schedule_generator.cpp
)

target_link_libraries(schedule_generator_test
    PRIVATE
        kpu_compiler
        Catch2::Catch2WithMain
)

target_include_directories(schedule_generator_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Register with CTest
add_test(NAME schedule_generator_test
    COMMAND schedule_generator_test
)

set_target_properties(schedule_generator_test PROPERTIES
    FOLDER "Tests/Compiler"
    LABELS "unit;compiler"
)
