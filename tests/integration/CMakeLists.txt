# ============================================================================
# tests/integration/CMakeLists.txt
# Integration tests

if(KPU_BUILD_TESTS)
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
endif()

if(KPU_BUILD_PYTHON_BINDINGS AND KPU_BUILD_TESTS)

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

    # On Windows, add Python and build DLL directories to PATH for the test
    if(WIN32)
        # Get Python installation directory (where pythonXX.dll lives)
        get_filename_component(PYTHON_DIR "${Python_EXECUTABLE}" DIRECTORY)
        file(TO_NATIVE_PATH "${PYTHON_DIR}" PYTHON_DIR_NATIVE)

        # Get build output directory (for any project DLLs)
        file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/$<CONFIG>" BUILD_BIN_NATIVE)

        # Start building PATH with Python directory
        set(TEST_PATH "${PYTHON_DIR_NATIVE}")

        # Add Python DLLs subdirectory if it exists (some Python installations)
        if(EXISTS "${PYTHON_DIR}/DLLs")
            file(TO_NATIVE_PATH "${PYTHON_DIR}/DLLs" PYTHON_DLLS_NATIVE)
            set(TEST_PATH "${TEST_PATH};${PYTHON_DLLS_NATIVE}")
        endif()

        # Add build output directory
        set(TEST_PATH "${TEST_PATH};${BUILD_BIN_NATIVE}")

        # Add system PATH last
        set(TEST_PATH "${TEST_PATH};$ENV{PATH}")

        # Set the PATH environment variable
        # This is required for pybind11::embed to find pythonXX.dll at runtime
        set_tests_properties(python_cpp_integration PROPERTIES
            ENVIRONMENT "PATH=${TEST_PATH}"
        )

        message(STATUS "Python directory for test: ${PYTHON_DIR_NATIVE}")
    endif()
endif()
