# ============================================================================
# src/bindings/c/CMakeLists.txt
# C API bindings

set(KPU_C_API_SOURCES
    kpu_c_api.cpp
)

set(KPU_C_API_HEADERS
    kpu_c_api.h
)

add_library(kpu_c_api SHARED ${KPU_C_API_SOURCES} ${KPU_C_API_HEADERS})
add_library(StillwaterKPU::C_API ALIAS kpu_c_api)

target_include_directories(kpu_c_api
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kpu_c_api
    PUBLIC
        StillwaterKPU::Simulator
)

kpu_set_target_options(kpu_c_api)

set_target_properties(kpu_c_api PROPERTIES
    OUTPUT_NAME stillwater_kpu_c_api
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    FOLDER "src/Bindings"
    PUBLIC_HEADER "${KPU_C_API_HEADERS}"
)

# Export C symbols
set_target_properties(kpu_c_api PROPERTIES
    C_VISIBILITY_PRESET default
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

install(TARGETS kpu_c_api
    EXPORT StillwaterKPUTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Development
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/stillwater/kpu
        COMPONENT Development
)

