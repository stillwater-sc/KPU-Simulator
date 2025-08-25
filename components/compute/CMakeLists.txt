# ============================================================================
# components/compute/CMakeLists.txt
# Compute subsystem component

set(KPU_COMPUTE_SOURCES
    src/matrix_unit.cpp
    src/vector_unit.cpp
    src/scalar_unit.cpp
    src/activation_unit.cpp
    src/instruction_decoder.cpp
)

set(KPU_COMPUTE_HEADERS
    include/stillwater/kpu/compute/compute_engine.hpp
    include/stillwater/kpu/compute/matrix_unit.hpp
    include/stillwater/kpu/compute/vector_unit.hpp
    include/stillwater/kpu/compute/scalar_unit.hpp
    include/stillwater/kpu/compute/activation_unit.hpp
    include/stillwater/kpu/compute/data_format.hpp
    include/stillwater/kpu/compute/instruction_decoder.hpp
)

add_library(kpu_compute ${KPU_COMPUTE_SOURCES} ${KPU_COMPUTE_HEADERS})
add_library(StillwaterKPU::Compute ALIAS kpu_compute)

target_include_directories(kpu_compute
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(kpu_compute
    PUBLIC
        StillwaterKPU::Memory
        Threads::Threads
        fmt::fmt
        spdlog::spdlog
)

# CUDA support
if(KPU_ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_sources(kpu_compute PRIVATE src/cuda_matrix_unit.cu)
    target_link_libraries(kpu_compute PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_definitions(kpu_compute PUBLIC KPU_HAS_CUDA)
endif()

# OpenCL support
if(KPU_ENABLE_OPENCL AND OpenCL_FOUND)
    target_sources(kpu_compute PRIVATE src/opencl_matrix_unit.cpp)
    target_link_libraries(kpu_compute PUBLIC OpenCL::OpenCL)
    target_compile_definitions(kpu_compute PUBLIC KPU_HAS_OPENCL)
endif()

kpu_set_target_options(kpu_compute)

set_target_properties(kpu_compute PROPERTIES
    OUTPUT_NAME stillwater_kpu_compute
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    FOLDER "Components"
    PUBLIC_HEADER "${KPU_COMPUTE_HEADERS}"
)

if(KPU_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(kpu_compute PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(kpu_compute PUBLIC KPU_HAS_OPENMP)
endif()

install(TARGETS kpu_compute
    EXPORT StillwaterKPUTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT Development
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/stillwater/kpu/compute
        COMPONENT Development
)

if(KPU_BUILD_TESTS)
    add_subdirectory(tests)
endif()
