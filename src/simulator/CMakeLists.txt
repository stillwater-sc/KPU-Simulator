# ============================================================================
# src/simulator/CMakeLists.txt
# Main simulator implementation

set(KPU_SIMULATOR_SOURCES
    kpu_simulator.cpp
    device_manager.cpp
    instruction_set.cpp
    execution_engine.cpp
    trace_logger.cpp
)

set(KPU_SIMULATOR_HEADERS
    kpu_simulator.hpp
    device_manager.hpp
    instruction_set.hpp
    execution_engine.hpp
    trace_logger.hpp
)

add_library(kpu_simulator ${KPU_SIMULATOR_SOURCES} ${KPU_SIMULATOR_HEADERS})
add_library(StillwaterKPU::Simulator ALIAS kpu_simulator)

target_include_directories(kpu_simulator
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kpu_simulator
    PUBLIC
        StillwaterKPU::Components
        Threads::Threads
        fmt::fmt
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

kpu_set_target_options(kpu_simulator)

set_target_properties(kpu_simulator PROPERTIES
    OUTPUT_NAME stillwater_kpu_simulator
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    FOLDER "Core"
)

install(TARGETS kpu_simulator
    EXPORT StillwaterKPUTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Development
)