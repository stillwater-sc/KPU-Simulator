 ============================================================================
# components/memory/CMakeLists.txt
# Memory subsystem component

# Memory component library
set(KPU_MEMORY_SOURCES
    src/main_memory.cpp
    src/scratchpad.cpp
    src/cache.cpp
    src/memory_controller.cpp
    src/address_translation.cpp
)

set(KPU_MEMORY_HEADERS
    include/stillwater/kpu/memory/memory_interface.hpp
    include/stillwater/kpu/memory/main_memory.hpp
    include/stillwater/kpu/memory/scratchpad.hpp
    include/stillwater/kpu/memory/cache.hpp
    include/stillwater/kpu/memory/memory_controller.hpp
    include/stillwater/kpu/memory/address_translation.hpp
)

# Create memory library
add_library(kpu_memory ${KPU_MEMORY_SOURCES} ${KPU_MEMORY_HEADERS})
add_library(StillwaterKPU::Memory ALIAS kpu_memory)

# Configure target
target_include_directories(kpu_memory
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(kpu_memory
    PUBLIC
        Threads::Threads
        fmt::fmt
        spdlog::spdlog
)

# Apply compiler options
kpu_set_target_options(kpu_memory)

# Set properties
set_target_properties(kpu_memory PROPERTIES
    OUTPUT_NAME stillwater_kpu_memory
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    FOLDER "Components"
    PUBLIC_HEADER "${KPU_MEMORY_HEADERS}"
)

# Enable OpenMP if available
if(KPU_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(kpu_memory PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(kpu_memory PUBLIC KPU_HAS_OPENMP)
endif()

# Install library and headers
install(TARGETS kpu_memory
    EXPORT StillwaterKPUTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Development
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/stillwater/kpu/memory
        COMPONENT Development
)

# Tests
if(KPU_BUILD_TESTS)
    add_subdirectory(tests)
endif()
