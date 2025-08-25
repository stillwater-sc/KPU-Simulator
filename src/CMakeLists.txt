# ============================================================================
# src/CMakeLists.txt
# Core simulator engine

add_subdirectory(simulator)

if(KPU_BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()