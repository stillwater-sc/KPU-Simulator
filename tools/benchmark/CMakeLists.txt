# ============================================================================
# tools/benchmark/CMakeLists.txt
# KPU Benchmark Tools
# ============================================================================

# KPU Benchmark Suite
kpu_add_tool(benchmark
    FOLDER "Benchmark"
    DESCRIPTION "KPU benchmark suite"
    SOURCES
        kpu-benchmark/benchmark.cpp
        kpu-benchmark/matrix_benchmarks.cpp
        kpu-benchmark/memory_benchmarks.cpp
)
