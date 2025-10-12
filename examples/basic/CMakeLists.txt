# ============================================================================
# examples/basic/CMakeLists.txt
# Basic examples

function(kpu_add_example name)
    set(options "")
    set(oneValueArgs FOLDER DESCRIPTION)
    set(multiValueArgs SOURCES LIBS)
    cmake_parse_arguments(EXAMPLE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    add_executable(example_${name} ${EXAMPLE_SOURCES})
    
    target_link_libraries(example_${name} PRIVATE
        StillwaterKPU::Simulator
        ${EXAMPLE_LIBS}
    )
    
    kpu_set_target_options(example_${name})
    
    set_target_properties(example_${name} PROPERTIES
        FOLDER "Examples/${EXAMPLE_FOLDER}"
    )
    
    install(TARGETS example_${name}
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/examples/bin
        COMPONENT Examples
    )
    
    # Install source code for reference
    install(FILES ${EXAMPLE_SOURCES}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/examples/${EXAMPLE_FOLDER}
        COMPONENT Examples
    )
endfunction()

kpu_add_example(hello_kpu
    FOLDER "basic"
    DESCRIPTION "Simple first KPU program"
    SOURCES hello_kpu.cpp
)

kpu_add_example(data_movement_pipeline
    FOLDER "basic"
    DESCRIPTION "Full data movement pipeline"
    SOURCES data_movement_pipeline.cpp
)

kpu_add_example(matrix_multiply
    FOLDER "basic"
    DESCRIPTION "Basic matrix multiplication"
    SOURCES matrix_multiply.cpp
)

kpu_add_example(memory_management
    FOLDER "basic"
    DESCRIPTION "Memory usage patterns"
    SOURCES memory_management.cpp
)

kpu_add_example(system_config_demo
    FOLDER "basic"
    DESCRIPTION "JSON-based system configuration"
    SOURCES system_config_demo.cpp
)

# Link system config demo against system library
target_link_libraries(example_system_config_demo PRIVATE
    StillwaterKPU::System
)
