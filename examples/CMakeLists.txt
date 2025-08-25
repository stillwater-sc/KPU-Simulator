# ============================================================================
# examples/CMakeLists.txt
# Example applications

add_subdirectory(basic)
add_subdirectory(advanced)

if(KPU_BUILD_PYTHON_BINDINGS)
    # Python examples are handled by the Python package
    install(DIRECTORY python/
        DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/examples/python
        COMPONENT Examples
    )
endif()

add_subdirectory(tutorials)

