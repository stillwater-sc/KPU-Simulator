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