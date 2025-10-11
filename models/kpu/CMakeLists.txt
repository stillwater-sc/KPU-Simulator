# ============================================================================
# models/kpu/CMakeLists.txt
# Different Host+KPU models

function(add_model name)
    set(options "")
    set(oneValueArgs FOLDER DESCRIPTION)
    set(multiValueArgs SOURCES LIBS)
    cmake_parse_arguments(MODEL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    add_executable(model_${name} ${MODEL_SOURCES})
    
    target_link_libraries(model_${name} PRIVATE
        StillwaterKPU::Simulator
	    ${MODEL_LIBS}
    )
    
    kpu_set_target_options(model_${name})
    
    set_target_properties(model_${name} PROPERTIES
        FOLDER "Models/${MODEL_FOLDER}"
    )
    
    install(TARGETS model_${name}
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/models/bin
	    COMPONENT Models
    )
    
    # Install source code for reference
    install(FILES ${MODEL_SOURCES}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/models/${MODEL_FOLDER}
	    COMPONENT Models
    )
endfunction()

add_model(host_t100
    FOLDER "kpu"
    DESCRIPTION "host + T100 KPU simulator"
    SOURCES host_t100.cpp
)

# Link targets against system library to include system configuration features
target_link_libraries(model_host_t100 PRIVATE
    StillwaterKPU::System
)

add_model(host_t100_autonomous
    FOLDER "kpu"
    DESCRIPTION "host + T100 KPU simulator with autonomous component orchestration"
    SOURCES host_t100_autonomous.cpp
)

# Link autonomous model against system library
target_link_libraries(model_host_t100_autonomous PRIVATE
    StillwaterKPU::System
)

# Add include directory for autonomous_orchestrator.hpp
target_include_directories(model_host_t100_autonomous PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_model(test_debug_dataflow
    FOLDER "kpu"
    DESCRIPTION "Debug test for data flow through KPU pipeline"
    SOURCES test_debug_dataflow.cpp
)

target_link_libraries(model_test_debug_dataflow PRIVATE
    StillwaterKPU::System
)

add_model(matmul_tiled_autonomous
    FOLDER "kpu"
    DESCRIPTION "Configurable tiled matrix multiplication with autonomous orchestration"
    SOURCES matmul_tiled_autonomous.cpp
)

target_link_libraries(model_matmul_tiled_autonomous PRIVATE
    StillwaterKPU::System
)

# Add include directory for autonomous_orchestrator.hpp
target_include_directories(model_matmul_tiled_autonomous PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)