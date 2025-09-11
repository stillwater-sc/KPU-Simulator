# ============================================================================
# tests/integration/CMakeLists.txt
# Integration tests

if(LATER)
kpu_add_test(
    NAME end_to_end
    INTEGRATION
    SOURCES test_end_to_end.cpp
    LIBS StillwaterKPU::Simulator
)

kpu_add_test(
    NAME multi_component
    INTEGRATION
    SOURCES test_multi_component.cpp
    LIBS StillwaterKPU::Simulator
)

if(KPU_BUILD_PYTHON_BINDINGS)
    kpu_add_test(
        NAME python_cpp_integration
        INTEGRATION
        SOURCES test_python_cpp_integration.cpp
        LIBS StillwaterKPU::Simulator
    )
endif()

endif()
