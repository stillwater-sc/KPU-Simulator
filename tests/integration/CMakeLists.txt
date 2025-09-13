# ============================================================================
# tests/integration/CMakeLists.txt
# Integration tests

kpu_add_test(
    NAME end_to_end
    INTEGRATION
    SOURCES test_end_to_end.cpp
    LIBS StillwaterKPU::Simulator StillwaterKPU::System StillwaterKPU::Driver
)

kpu_add_test(
    NAME multi_component
    INTEGRATION
    SOURCES test_multi_component.cpp
    LIBS StillwaterKPU::Simulator StillwaterKPU::System StillwaterKPU::Driver
)

if(KPU_BUILD_PYTHON_BINDINGS)
    kpu_add_test(
        NAME python_cpp_integration
        INTEGRATION
        SOURCES test_python_cpp_integration.cpp
        LIBS StillwaterKPU::Simulator StillwaterKPU::System StillwaterKPU::Driver
    )
    
    # Add Python embedding support for the integration test
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    target_link_libraries(test_python_cpp_integration 
        PRIVATE 
        Python::Python
        pybind11::embed
    )
    
    # Configure paths where Python modules are built  
    set(PYTHON_MODULES_BUILD_DIR "${CMAKE_BINARY_DIR}/src/bindings/python")
    set(PYTHON_MODULES_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/bindings/python")
    # On multi-config generators (Visual Studio), modules go into config subdirectories
    set(PYTHON_MODULES_BUILD_DIR_DEBUG "${CMAKE_BINARY_DIR}/src/bindings/python/Debug")
    set(PYTHON_MODULES_BUILD_DIR_RELEASE "${CMAKE_BINARY_DIR}/src/bindings/python/Release")
    
    target_compile_definitions(test_python_cpp_integration 
        PRIVATE 
        KPU_BUILD_PYTHON_BINDINGS
        PYTHON_MODULES_BUILD_DIR="${PYTHON_MODULES_BUILD_DIR}"
        PYTHON_MODULES_SOURCE_DIR="${PYTHON_MODULES_SOURCE_DIR}"
        PYTHON_MODULES_BUILD_DIR_DEBUG="${PYTHON_MODULES_BUILD_DIR_DEBUG}"
        PYTHON_MODULES_BUILD_DIR_RELEASE="${PYTHON_MODULES_BUILD_DIR_RELEASE}"
    )
    
    # Ensure Python module is built before running the test
    add_dependencies(test_python_cpp_integration stillwater_toplevel)
endif()
