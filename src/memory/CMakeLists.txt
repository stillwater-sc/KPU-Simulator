# Memory module - manages its own sources and targets

set(MEMORY_SOURCES
    scratchpad.cpp
    cache.cpp
)

# Create memory library
add_library(kpu_memory STATIC ${MEMORY_SOURCES})

# Create the alias that other parts expect
add_library(StillwaterKPU::Memory ALIAS kpu_memory)

target_include_directories(kpu_memory 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Apply common target options
kpu_set_target_options(kpu_memory)

set_target_properties(kpu_memory PROPERTIES
    EXPORT_NAME Memory
    FOLDER "Core"
    POSITION_INDEPENDENT_CODE ON  # Enable PIC for shared library compatibility
)

# Don't export for now to avoid dependency chain issues
install(TARGETS kpu_memory
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
