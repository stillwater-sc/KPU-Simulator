cmake_minimum_required(VERSION 3.16)
project(kpu_simulator VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # Remove problematic default flags
    foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE 
                     CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REGEX REPLACE "/RTC[^ ]*" "" ${flag_var} "${${flag_var}}")
        string(REGEX REPLACE "/W[0-4]" "" ${flag_var} "${${flag_var}}")
    endforeach()
    
    # Set our own flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /EHsc")
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Od /D_DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /O2 /Zi /DNDEBUG")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "/MD /O1 /DNDEBUG")
    
    # Define for DLL export/import
    add_definitions(-DBUILDING_KPU_SIMULATOR)
    
    # Suppress specific warnings
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_compile_options(/wd4251 /wd4275 /wd4996)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files for the core simulator
set(SIMULATOR_SOURCES
    src/simulator/kpu_simulator.cpp
)

# Header files
set(SIMULATOR_HEADERS
    include/sw/kpu/simulator.hpp
)

# Create the core simulator library
add_library(kpu_simulator_core STATIC ${SIMULATOR_SOURCES} ${SIMULATOR_HEADERS})
target_include_directories(kpu_simulator_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Test executable
add_executable(kpu_test tests/test_main.cpp)
target_link_libraries(kpu_test kpu_simulator_core)

# Python bindings (optional, if pybind11 is available)
find_package(pybind11 QUIET)
if(pybind11_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/src/bindings/python/kpu_bindings.cpp")
    message(STATUS "Building Python bindings with pybind11 ${pybind11_VERSION}")
    pybind11_add_module(stillwater_kpu src/bindings/python/kpu_bindings.cpp)
    target_link_libraries(stillwater_kpu PRIVATE kpu_simulator_core)
    
    # Set properties for the Python module
    target_compile_definitions(stillwater_kpu PRIVATE VERSION_INFO=${PROJECT_VERSION})
    
    # Install the Python module
    install(TARGETS stillwater_kpu
        COMPONENT python
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/python/stillwater_kpu
    )
    message(STATUS "Building Python bindings")
else()
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found. Install with: pip install 'pybind11[global]'")
        
        # Try to provide helpful debug info
        execute_process(
            COMMAND python --version
            OUTPUT_VARIABLE PYTHON_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(PYTHON_VERSION)
            message(STATUS "Python version: ${PYTHON_VERSION}")
        else()
            message(STATUS "Python not found in PATH")
        endif()
    endif()
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/src/bindings/python/kpu_bindings.cpp")
        message(STATUS "Python bindings source not found at src/bindings/python/kpu_bindings.cpp")
    endif()
    message(STATUS "Python bindings disabled")   
endif()

# Installation
install(TARGETS kpu_simulator_core
    EXPORT kpu_simulator_targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

if(TARGET kpu_test)
    install(TARGETS kpu_test DESTINATION bin)
endif()

# Export targets for use by other projects
install(EXPORT kpu_simulator_targets
    FILE kpu_simulator_targets.cmake
    NAMESPACE kpu::
    DESTINATION lib/cmake/kpu_simulator
)

# Create config file for find_package support
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    kpu_simulator_config_version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    cmake/kpu_simulator_config.cmake.in
    kpu_simulator_config.cmake
    INSTALL_DESTINATION lib/cmake/kpu_simulator
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/kpu_simulator_config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/kpu_simulator_config_version.cmake
    DESTINATION lib/cmake/kpu_simulator
)


# Print comprehensive build summary
message(STATUS "")
message(STATUS "========== KPU Simulator Build Configuration ==========")
message(STATUS "Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Generator: ${CMAKE_GENERATOR}")
if(MSVC)
    message(STATUS "MSVC Version: ${MSVC_VERSION}")
    message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
    message(STATUS "CXX Flags (${CMAKE_BUILD_TYPE}): ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
endif()
message(STATUS "Source files: ${SIMULATOR_SOURCES}")
if(TARGET kpu_test)
    message(STATUS "Test executable: ENABLED")
else()
    message(STATUS "Test executable: DISABLED")
endif()
if(TARGET stillwater_kpu)
    message(STATUS "Python bindings: ENABLED")
else()
    message(STATUS "Python bindings: DISABLED")
endif()
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "======================================================")
message(STATUS "")

# Helpful build instructions
message(STATUS "To build:")
message(STATUS "  cmake --build . --config ${CMAKE_BUILD_TYPE}")
message(STATUS "To run tests:")
message(STATUS "  .\\${CMAKE_BUILD_TYPE}\\kpu_test.exe")
message(STATUS "")