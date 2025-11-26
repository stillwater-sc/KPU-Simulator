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

# L2TileScheduler demonstration
add_executable(example_l2_tile_scheduler_demo
    l2_tile_scheduler_demo.cpp
)

target_link_libraries(example_l2_tile_scheduler_demo
    PRIVATE
        kpu_compiler
)

target_include_directories(example_l2_tile_scheduler_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(example_l2_tile_scheduler_demo PROPERTIES
    FOLDER "Examples/Compiler"
    OUTPUT_NAME l2_tile_scheduler_demo
)

# Installation
install(TARGETS example_l2_tile_scheduler_demo
    RUNTIME DESTINATION examples/compiler
)

# ScheduleCharacterizer demonstration
add_executable(example_schedule_characterizer_demo
    schedule_characterizer_demo.cpp
)

target_link_libraries(example_schedule_characterizer_demo
    PRIVATE
        kpu_compiler
)

target_include_directories(example_schedule_characterizer_demo
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(example_schedule_characterizer_demo PROPERTIES
    FOLDER "Examples/Compiler"
    OUTPUT_NAME schedule_characterizer_demo
)

# Installation
install(TARGETS example_schedule_characterizer_demo
    RUNTIME DESTINATION examples/compiler
)

# DataFlow strategy demonstration
add_executable(dataflow_strategy
    dataflow_strategy.cpp
)

target_link_libraries(dataflow_strategy
    PRIVATE
        kpu_compiler
)

target_include_directories(dataflow_strategy
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(dataflow_strategy PROPERTIES
    FOLDER "Examples/Compiler"
    OUTPUT_NAME dataflow_strategy
)

# Installation
install(TARGETS dataflow_strategy
    RUNTIME DESTINATION examples/compiler
)

# HardwareDesignExplorer demonstration
add_executable(example_hardware_design_explorer
    hardware_design_explorer.cpp
)

target_link_libraries(example_hardware_design_explorer
    PRIVATE
        kpu_compiler
)

target_include_directories(example_hardware_design_explorer
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(example_hardware_design_explorer PROPERTIES
    FOLDER "Examples/Compiler"
    OUTPUT_NAME hardware_design_explorer
)

# Installation
install(TARGETS example_hardware_design_explorer
    RUNTIME DESTINATION examples/compiler
)

# L3OverfetchAnalyzer demonstration
add_executable(example_l3_overfetch_analyzer
    l3_overfetch_analyzer.cpp
)

target_link_libraries(example_l3_overfetch_analyzer
    PRIVATE
        kpu_compiler
)

target_include_directories(example_l3_overfetch_analyzer
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(example_l3_overfetch_analyzer PROPERTIES
    FOLDER "Examples/Compiler"
    OUTPUT_NAME l3_overfetch_analyzer
)

# Installation
install(TARGETS example_l3_overfetch_analyzer
    RUNTIME DESTINATION examples/compiler
)

# L3TransformerAnalysis demonstration
add_executable(example_l3_transformer_analysis
    l3_transformer_analysis.cpp
)

target_link_libraries(example_l3_transformer_analysis
    PRIVATE
        kpu_compiler
)

target_include_directories(example_l3_transformer_analysis
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(example_l3_transformer_analysis PROPERTIES
    FOLDER "Examples/Compiler"
    OUTPUT_NAME l3_transformer_analysis
)

# Installation
install(TARGETS example_l3_transformer_analysis
    RUNTIME DESTINATION examples/compiler
)

# L3ComprehensiveAnalysis demonstration
add_executable(example_l3_comprehensive_analysis
    l3_comprehensive_analysis.cpp
)

target_link_libraries(example_l3_comprehensive_analysis
    PRIVATE
        kpu_compiler
)

target_include_directories(example_l3_comprehensive_analysis
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(example_l3_comprehensive_analysis PROPERTIES
    FOLDER "Examples/Compiler"
    OUTPUT_NAME l3_comprehensive_analysis
)

# Installation
install(TARGETS example_l3_comprehensive_analysis
    RUNTIME DESTINATION examples/compiler
)

# L3FocusedAnalysis demonstration
add_executable(example_l3_focused_analysis
    l3_focused_analysis.cpp
)

target_link_libraries(example_l3_focused_analysis
    PRIVATE
        kpu_compiler
)

target_include_directories(example_l3_focused_analysis
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(example_l3_focused_analysis PROPERTIES
    FOLDER "Examples/Compiler"
    OUTPUT_NAME l3_focused_analysis
)

# Installation
install(TARGETS example_l3_focused_analysis
    RUNTIME DESTINATION examples/compiler
)
