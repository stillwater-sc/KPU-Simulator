# ============================================================================
# tests/CMakeLists.txt
# Test suite configuration

# Test categories
add_subdirectory(integration)
add_subdirectory(performance)
add_subdirectory(regression)

# Function to add test data
function(kpu_add_test_data)
    set(options "")
    set(oneValueArgs NAME DESTINATION)
    set(multiValueArgs FILES)
    cmake_parse_arguments(DATA "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    foreach(file ${DATA_FILES})
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/data/${file}
            ${CMAKE_CURRENT_BINARY_DIR}/${DATA_DESTINATION}/${file}
            COPYONLY
        )
    endforeach()
endfunction()

# Copy test data
kpu_add_test_data(
    NAME matrices
    DESTINATION data/matrices
    FILES
        small_matrix_a.dat
        small_matrix_b.dat
        large_matrix_a.dat
        large_matrix_b.dat
)
