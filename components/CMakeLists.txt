# ============================================================================
# components/CMakeLists.txt
# Hardware component libraries root

# Component library configuration
set(KPU_COMPONENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Export component include directory for other components
set(KPU_COMPONENT_INCLUDE_DIR ${KPU_COMPONENT_INCLUDE_DIR} PARENT_SCOPE)

# Component subdirectories
add_subdirectory(memory)
add_subdirectory(compute)
add_subdirectory(fabric)
add_subdirectory(dma)
add_subdirectory(power)

# Create umbrella target for all components
add_library(kpu_components INTERFACE)
add_library(StillwaterKPU::Components ALIAS kpu_components)

target_link_libraries(kpu_components INTERFACE
    StillwaterKPU::Memory
    StillwaterKPU::Compute
    StillwaterKPU::Fabric
    StillwaterKPU::DMA
    StillwaterKPU::Power
)

# Install component umbrella target
install(TARGETS kpu_components
    EXPORT StillwaterKPUTargets
    COMPONENT Development
)

