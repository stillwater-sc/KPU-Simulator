# ============================================================================
# tools/python/CMakeLists.txt
# Python development tools
#
# These are Python scripts that don't need to be built, but we include
# this CMakeLists.txt for completeness.
# ============================================================================

# Python tools are not compiled - they are run directly with Python
# This file exists to satisfy the CMake add_subdirectory() call

# Optionally install Python tools
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION share/kpu/tools/python
    COMPONENT Tools
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
)
