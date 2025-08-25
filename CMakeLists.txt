cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project definition
project(StillwaterKPU 
    VERSION 1.0.0
    DESCRIPTION "Stillwater Knowledge Processing Unit Simulator"
    HOMEPAGE_URL "https://github.com/stillwater-sc/kpu-simulator"
    LANGUAGES CXX C
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerOptions)
include(Dependencies)
include(Documentation)
include(Testing)
include(Packaging)

# Project options
option(KPU_BUILD_TESTS "Build test suite" ON)
option(KPU_BUILD_EXAMPLES "Build examples" ON)
option(KPU_BUILD_TOOLS "Build development tools" ON)
option(KPU_BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(KPU_BUILD_BENCHMARKS "Build benchmark suite" ON)
option(KPU_BUILD_DOCS "Build documentation" OFF)
option(KPU_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(KPU_ENABLE_CUDA "Enable CUDA support" OFF)
option(KPU_ENABLE_OPENCL "Enable OpenCL support" OFF)
option(KPU_ENABLE_PROFILING "Enable profiling support" OFF)
option(KPU_ENABLE_SANITIZERS "Enable sanitizers" OFF)
option(KPU_STATIC_ANALYSIS "Enable static analysis" OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

# Global properties
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake Targets")

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Platform-specific settings
if(WIN32)
    add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Find required dependencies
find_package(Threads REQUIRED)

# Optional dependencies
if(KPU_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found, disabling parallel operations")
        set(KPU_ENABLE_OPENMP OFF)
    endif()
endif()

if(KPU_ENABLE_CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        enable_language(CUDA)
    else()
        message(WARNING "CUDA not found, disabling CUDA support")
        set(KPU_ENABLE_CUDA OFF)
    endif()
endif()

if(KPU_ENABLE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL found: ${OpenCL_VERSION_STRING}")
    else()
        message(WARNING "OpenCL not found, disabling OpenCL support")
        set(KPU_ENABLE_OPENCL OFF)
    endif()
endif()

# Third-party dependencies
add_subdirectory(third_party)

# Component libraries (hardware components)
add_subdirectory(components)

# Core simulator engine
add_subdirectory(src)

# Development tools
if(KPU_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# Examples
if(KPU_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(KPU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(KPU_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Documentation
if(KPU_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Packaging
add_subdirectory(packaging)

# Installation
include(GNUInstallDirs)

# Install public headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Install CMake config files
include(CMakePackageConfigHelpers)

# Generate package config
configure_package_config_file(
    cmake/StillwaterKPUConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/StillwaterKPUConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StillwaterKPU
)

# Generate version file
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/StillwaterKPUConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install package config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/StillwaterKPUConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/StillwaterKPUConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StillwaterKPU
)

# Export targets for build tree
export(EXPORT StillwaterKPUTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/StillwaterKPUTargets.cmake
    NAMESPACE StillwaterKPU::
)

# Export targets for install tree
install(EXPORT StillwaterKPUTargets
    FILE StillwaterKPUTargets.cmake
    NAMESPACE StillwaterKPU::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StillwaterKPU
)

# Register package in user registry
export(PACKAGE StillwaterKPU)

# Summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project: ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  Tests: ${KPU_BUILD_TESTS}")
message(STATUS "  Examples: ${KPU_BUILD_EXAMPLES}")
message(STATUS "  Tools: ${KPU_BUILD_TOOLS}")
message(STATUS "  Python bindings: ${KPU_BUILD_PYTHON_BINDINGS}")
message(STATUS "  Benchmarks: ${KPU_BUILD_BENCHMARKS}")
message(STATUS "  Documentation: ${KPU_BUILD_DOCS}")
message(STATUS "")
message(STATUS "Acceleration:")
message(STATUS "  OpenMP: ${KPU_ENABLE_OPENMP}")
message(STATUS "  CUDA: ${KPU_ENABLE_CUDA}")
message(STATUS "  OpenCL: ${KPU_ENABLE_OPENCL}")
message(STATUS "")
message(STATUS "Development:")
message(STATUS "  Profiling: ${KPU_ENABLE_PROFILING}")
message(STATUS "  Sanitizers: ${KPU_ENABLE_SANITIZERS}")
message(STATUS "  Static analysis: ${KPU_STATIC_ANALYSIS}")
message(STATUS "")