// include/sw/kpu/kpu.hpp
// Main public header for Stillwater KPU Simulator

#pragma once

#define STILLWATER_KPU_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
#define STILLWATER_KPU_VERSION_MINOR @PROJECT_VERSION_MINOR@
#define STILLWATER_KPU_VERSION_PATCH @PROJECT_VERSION_PATCH@
#define STILLWATER_KPU_VERSION "@PROJECT_VERSION@"

// Feature detection macros
#ifdef KPU_HAS_OPENMP
#define STILLWATER_KPU_HAS_OPENMP 1
#else
#define STILLWATER_KPU_HAS_OPENMP 0
#endif

#ifdef KPU_HAS_CUDA
#define STILLWATER_KPU_HAS_CUDA 1
#else
#define STILLWATER_KPU_HAS_CUDA 0
#endif

#ifdef KPU_HAS_OPENCL
#define STILLWATER_KPU_HAS_OPENCL 1
#else
#define STILLWATER_KPU_HAS_OPENCL 0
#endif

// Core simulator
#include <sw/kpu/simulator/kpu_simulator.hpp>
#include <sw/kpu/simulator/device_manager.hpp>
#include <sw/kpu/simulator/instruction_set.hpp>

// Memory subsystem
#include <sw/kpu/memory/memory_interface.hpp>
#include <sw/kpu/memory/main_memory.hpp>
#include <sw/kpu/memory/scratchpad.hpp>
#include <sw/kpu/memory/cache.hpp>
#include <sw/kpu/memory/memory_controller.hpp>

// Compute subsystem
#include <sw/kpu/compute/compute_engine.hpp>
#include <sw/kpu/compute/matrix_unit.hpp>
#include <sw/kpu/compute/vector_unit.hpp>
#include <sw/kpu/compute/scalar_unit.hpp>
#include <sw/kpu/compute/data_format.hpp>

// Interconnect fabric
#include <sw/kpu/fabric/fabric_interface.hpp>
#include <sw/kpu/fabric/crossbar.hpp>
#include <sw/kpu/fabric/mesh_noc.hpp>
#include <sw/kpu/fabric/packet.hpp>

// DMA subsystem
#include <sw/kpu/dma/dma_engine.hpp>
#include <sw/kpu/dma/scatter_gather.hpp>
#include <sw/kpu/dma/streaming_dma.hpp>

// Utilities
#include <sw/kpu/utilities/logging.hpp>
#include <sw/kpu/utilities/profiler.hpp>
#include <sw/kpu/utilities/configuration.hpp>

namespace sw {
namespace kpu {

/**
 * @brief Get KPU simulator version string
 * @return Version string in format "major.minor.patch"
 */
inline const char* version() {
    return STILLWATER_KPU_VERSION;
}

/**
 * @brief Get KPU simulator version components
 * @param major Major version number
 * @param minor Minor version number  
 * @param patch Patch version number
 */
inline void version(int& major, int& minor, int& patch) {
    major = STILLWATER_KPU_VERSION_MAJOR;
    minor = STILLWATER_KPU_VERSION_MINOR;
    patch = STILLWATER_KPU_VERSION_PATCH;
}

/**
 * @brief Check if feature is available
 * @param feature Feature name ("openmp", "cuda", "opencl")
 * @return True if feature is available
 */
inline bool has_feature(const std::string& feature) {
    if (feature == "openmp") return STILLWATER_KPU_HAS_OPENMP;
    if (feature == "cuda") return STILLWATER_KPU_HAS_CUDA;
    if (feature == "opencl") return STILLWATER_KPU_HAS_OPENCL;
    return false;
}

} // namespace kpu
} // namespace sw
