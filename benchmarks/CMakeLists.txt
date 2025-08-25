# ============================================================================
# benchmarks/CMakeLists.txt
# Benchmark suite

# Enable benchmark timing
find_package(benchmark QUIET)
if(benchmark_FOUND)
    message(STATUS "Google Benchmark found, enabling detailed benchmarks")
    set(KPU_HAS_GOOGLE_BENCHMARK TRUE)
else()
    message(STATUS "Google Benchmark not found, using simple timing")
    set(KPU_HAS_GOOGLE_BENCHMARK FALSE)
endif()

function(kpu_add_benchmark name)
    set(options "")
    set(oneValueArgs FOLDER DESCRIPTION)
    set(multiValueArgs SOURCES LIBS)
    cmake_parse_arguments(BENCH "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    add_executable(benchmark_${name} ${BENCH_SOURCES})
    
    target_link_libraries(benchmark_${name} PRIVATE
        StillwaterKPU::Simulator
        ${BENCH_LIBS}
    )
    
    if(KPU_HAS_GOOGLE_BENCHMARK)
        target_link_libraries(benchmark_${name} PRIVATE benchmark::benchmark)
        target_compile_definitions(benchmark_${name} PRIVATE KPU_HAS_GOOGLE_BENCHMARK)
    endif()
    
    kpu_set_target_options(benchmark_${name})
    
    set_target_properties(benchmark_${name} PROPERTIES
        FOLDER "Benchmarks/${BENCH_FOLDER}"
    )
    
    install(TARGETS benchmark_${name}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT Tools
    )
    
    # Add as test for CI
    add_test(NAME benchmark_${name}
        COMMAND benchmark_${name} --benchmark_min_time=0.1
    )
    set_tests_properties(benchmark_${name} PROPERTIES LABELS "benchmark")
endfunction()

add_subdirectory(mlperf)
add_subdirectory(rodinia)
add_subdirectory(custom)

