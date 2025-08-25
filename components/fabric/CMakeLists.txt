# ============================================================================
# components/fabric/CMakeLists.txt
# Interconnect fabric component

set(KPU_FABRIC_SOURCES
    src/crossbar.cpp
    src/mesh_noc.cpp
    src/ring_noc.cpp
    src/packet.cpp
    src/router.cpp
)

set(KPU_FABRIC_HEADERS
    include/stillwater/kpu/fabric/fabric_interface.hpp
    include/stillwater/kpu/fabric/crossbar.hpp
    include/stillwater/kpu/fabric/mesh_noc.hpp
    include/stillwater/kpu/fabric/ring_noc.hpp
    include/stillwater/kpu/fabric/packet.hpp
    include/stillwater/kpu/fabric/router.hpp
    include/stillwater/kpu/fabric/flow_control.hpp
)

add_library(kpu_fabric ${KPU_FABRIC_SOURCES} ${KPU_FABRIC_HEADERS})
add_library(StillwaterKPU::Fabric ALIAS kpu_fabric)

target_include_directories(kpu_fabric
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(kpu_fabric
    PUBLIC
        StillwaterKPU::Memory
        Threads::Threads
        fmt::fmt
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

kpu_set_target_options(kpu_fabric)

set_target_properties(kpu_fabric PROPERTIES
    OUTPUT_NAME stillwater_kpu_fabric
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    FOLDER "Components"
    PUBLIC_HEADER "${KPU_FABRIC_HEADERS}"
)

install(TARGETS kpu_fabric
    EXPORT StillwaterKPUTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Development
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/stillwater/kpu/fabric
        COMPONENT Development
)

if(KPU_BUILD_TESTS)
    add_subdirectory(tests)
endif()

