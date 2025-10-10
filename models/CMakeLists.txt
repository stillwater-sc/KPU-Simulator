# ============================================================================
# models/CMakeLists.txt
# Compute models

# Add existing compute model directories
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cpu)
    add_subdirectory(cpu)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gpu)
    add_subdirectory(gpu)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tpu)
    add_subdirectory(tpu)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/kpu)
    add_subdirectory(kpu)
endif()

if(KPU_BUILD_PYTHON_BINDINGS)
    # Python examples are handled by the Python package
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/python)
        install(DIRECTORY python/
            DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/models/python
	    COMPONENT Models
        )
    endif()
endif()

