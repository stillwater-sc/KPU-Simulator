# ============================================================================
# tools/analysis/CMakeLists.txt
# KPU Analysis and Profiling Tools
# ============================================================================

# KPU Profiler
kpu_add_tool(profiler
    FOLDER "Analysis"
    DESCRIPTION "KPU performance profiler"
    SOURCES
        kpu-profiler/profiler.cpp
)

# KPU Trace Analyzer
kpu_add_tool(trace-analyzer
    FOLDER "Analysis"
    DESCRIPTION "KPU trace analysis and visualization"
    SOURCES
        kpu-trace-analyzer/trace_analyzer.cpp
    LIBS
        nlohmann_json::nlohmann_json
)
