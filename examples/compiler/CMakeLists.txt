# examples/compiler/CMakeLists.txt
# Compiler examples

# TileOptimizer demonstration
add_executable(example_tile_optimizer_demo
    tile_optimizer_demo.cpp
)

target_link_libraries(example_tile_optimizer_demo
    PRIVATE
        kpu_compiler
)

target_include_directories(example_tile_optimizer_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(example_tile_optimizer_demo PROPERTIES
    FOLDER "Examples/Compiler"
    OUTPUT_NAME tile_optimizer_demo
)

# Installation
install(TARGETS example_tile_optimizer_demo
    RUNTIME DESTINATION examples/compiler
)

# ScheduleGenerator demonstration
add_executable(example_schedule_generator_demo
    schedule_generator_demo.cpp
)

target_link_libraries(example_schedule_generator_demo
    PRIVATE
        kpu_compiler
)

target_include_directories(example_schedule_generator_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(example_schedule_generator_demo PROPERTIES
    FOLDER "Examples/Compiler"
    OUTPUT_NAME schedule_generator_demo
)

# Installation
install(TARGETS example_schedule_generator_demo
    RUNTIME DESTINATION examples/compiler
)
