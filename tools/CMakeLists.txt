# ============================================================================
# tools/CMakeLists.txt
# Development tools and utilities

# C++ tools
add_subdirectory(cpp)

# Python tools
if(KPU_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()