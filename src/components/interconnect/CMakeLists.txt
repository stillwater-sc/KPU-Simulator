# Interconnect Component Library
# Network-on-chip and data movement components

# Interconnect component sources
set(INTERCONNECT_COMPONENT_SOURCES
    dma_engine.cpp
    block_mover.cpp
)

# Create the interconnect components library
add_library(kpu_interconnect_components ${INTERCONNECT_COMPONENT_SOURCES})

# Set up include directories
target_include_directories(kpu_interconnect_components
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(kpu_interconnect_components
    PUBLIC
        kpu_memory_components  # Interconnect components depend on memory components
)

# Set target properties
set_target_properties(kpu_interconnect_components PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "kpu_interconnect_components"
    FOLDER "src/Components/Interconnect"
)

# Add compiler-specific options
if(MSVC)
    target_compile_definitions(kpu_interconnect_components PRIVATE BUILDING_KPU_SIMULATOR)
    target_compile_options(kpu_interconnect_components PRIVATE /W4)
else()
    target_compile_options(kpu_interconnect_components PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Export for installation
install(TARGETS kpu_interconnect_components
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)