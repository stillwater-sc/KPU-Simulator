# ============================================================================
# tools/cpp/CMakeLists.txt
# C++ development tools

# Function to create a tool executable
function(kpu_add_tool name)
    set(options "")
    set(oneValueArgs FOLDER DESCRIPTION)
    set(multiValueArgs SOURCES LIBS)
    cmake_parse_arguments(TOOL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    add_executable(${name} ${TOOL_SOURCES})
    add_executable(StillwaterKPU::${name} ALIAS ${name})
    
    target_link_libraries(${name} PRIVATE
        StillwaterKPU::Simulator
        ${TOOL_LIBS}
    )
    
    kpu_set_target_options(${name})
    
    set_target_properties(${name} PROPERTIES
        OUTPUT_NAME kpu-${name}
        FOLDER "Tools/${TOOL_FOLDER}"
    )
    
    install(TARGETS ${name}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT Tools
    )
endfunction()

# Configuration tool
kpu_add_tool(config
    FOLDER "Configuration"
    DESCRIPTION "KPU configuration and validation tool"
    SOURCES kpu-config/kpu_config.cpp
    LIBS nlohmann_json::nlohmann_json
)

# Profiler tool
kpu_add_tool(profiler
    FOLDER "Analysis"
    DESCRIPTION "KPU performance profiler"
    SOURCES 
        kpu-profiler/profiler.cpp
        kpu-profiler/profiler.hpp
)

# Assembler tool
kpu_add_tool(assembler
    FOLDER "Development"
    DESCRIPTION "KPU assembly language assembler"
    SOURCES
        kpu-assembler/assembler.cpp
        kpu-assembler/parser.hpp
        kpu-assembler/instruction_set.hpp
)

# Debugger tool
kpu_add_tool(debugger
    FOLDER "Development"
    DESCRIPTION "Interactive KPU debugger"
    SOURCES
        kpu-debugger/debugger.cpp
        kpu-debugger/breakpoint.hpp
        kpu-debugger/symbol_table.hpp
)

# Benchmark tool
kpu_add_tool(benchmark
    FOLDER "Performance"
    DESCRIPTION "KPU benchmark suite"
    SOURCES
        kpu-benchmark/benchmark.cpp
        kpu-benchmark/matrix_benchmarks.cpp
        kpu-benchmark/memory_benchmarks.cpp
)
