# ============================================================================
# tests/system/CMakeLists.txt
# System orchestration tests

add_kpu_test(test_toplevel_simulator "system" test_toplevel_simulator.cpp)

# Link against the system library
target_link_libraries(test_toplevel_simulator 
    PRIVATE 
    StillwaterKPU::System
)