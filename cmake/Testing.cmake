# cmake/Testing.cmake
# Testing configuration and utilities

if(KPU_BUILD_TESTS)
    include(CTest)
    
    # Test configuration
    set(KPU_TEST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/test_results)
    file(MAKE_DIRECTORY ${KPU_TEST_OUTPUT_DIR})
    
    # Function to add a test
    function(kpu_add_test)
        set(options UNIT INTEGRATION PERFORMANCE)
        set(oneValueArgs NAME TARGET WORKING_DIRECTORY)
        set(multiValueArgs SOURCES LIBS ARGS)
        cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
        
        if(NOT TEST_NAME)
            message(FATAL_ERROR "kpu_add_test: NAME is required")
        endif()
        
        if(TEST_SOURCES)
            # Create test executable
            if(NOT TEST_TARGET)
                set(TEST_TARGET test_${TEST_NAME})
            endif()
            
            add_executable(${TEST_TARGET} ${TEST_SOURCES})
            
            # Link libraries
            target_link_libraries(${TEST_TARGET} PRIVATE Catch2::Catch2WithMain)
            if(TEST_LIBS)
                target_link_libraries(${TEST_TARGET} PRIVATE ${TEST_LIBS})
            endif()
            
            # Apply compiler options
            kpu_set_target_options(${TEST_TARGET})
            
            # Set folder
            set_target_properties(${TEST_TARGET} PROPERTIES
                FOLDER "Tests"
            )
        endif()
        
        # Add to CTest
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_TARGET} ${TEST_ARGS})
        
        # Set working directory
        if(TEST_WORKING_DIRECTORY)
            set_tests_properties(${TEST_NAME} PROPERTIES
                WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY}
            )
        endif()
        
        # Set test properties based on type
        if(TEST_UNIT)
            set_tests_properties(${TEST_NAME} PROPERTIES LABELS "unit")
        elseif(TEST_INTEGRATION)
            set_tests_properties(${TEST_NAME} PROPERTIES LABELS "integration")
        elseif(TEST_PERFORMANCE)
            set_tests_properties(${TEST_NAME} PROPERTIES LABELS "performance")
        endif()
    endfunction()
    
    # Coverage support
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        option(KPU_ENABLE_COVERAGE "Enable code coverage" OFF)
        if(KPU_ENABLE_COVERAGE)
            add_compile_options(--coverage -O0 -g)
            add_link_options(--coverage)
        endif()
    endif()
endif()

