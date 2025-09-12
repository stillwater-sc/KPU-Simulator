# ============================================================================
# src/CMakeLists.txt
# Core simulator engine

add_subdirectory(system)
add_subdirectory(simulator)
add_subdirectory(driver)
add_subdirectory(memory)

# TODO: Add when implemented
# add_subdirectory(dma)
# add_subdirectory(block_mover) 
# add_subdirectory(streamer)

if(KPU_BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()
