#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Opaque handle types
typedef struct KPUSimulator* KPUHandle;

// Error codes
typedef enum {
    KPU_SUCCESS = 0,
    KPU_ERROR_INVALID_HANDLE = -1,
    KPU_ERROR_OUT_OF_BOUNDS = -2,
    KPU_ERROR_INVALID_DIMENSIONS = -3,
    KPU_ERROR_NULL_POINTER = -4,
    KPU_ERROR_UNKNOWN = -5
} KPUError;

// Matrix dimensions structure
typedef struct {
    uint32_t rows;
    uint32_t cols;
} KPUMatrixDim;

// KPU Simulator lifecycle
KPUHandle kpu_create(uint64_t main_memory_size, uint64_t scratchpad_size);
void kpu_destroy(KPUHandle handle);

// Memory operations
KPUError kpu_main_memory_read(KPUHandle handle, uint64_t addr, void* data, uint64_t size);
KPUError kpu_main_memory_write(KPUHandle handle, uint64_t addr, const void* data, uint64_t size);
KPUError kpu_scratchpad_read(KPUHandle handle, uint64_t addr, void* data, uint64_t size);
KPUError kpu_scratchpad_write(KPUHandle handle, uint64_t addr, const void* data, uint64_t size);

uint64_t kpu_main_memory_size(KPUHandle handle);
uint64_t kpu_scratchpad_size(KPUHandle handle);

// DMA operations
KPUError kpu_dma_transfer_sync(KPUHandle handle,
                              uint64_t src_addr, uint64_t dst_addr, uint64_t size,
                              bool src_is_main_memory, bool dst_is_main_memory);

// Compute operations
KPUError kpu_matmul_f32(KPUHandle handle,
                       uint64_t addr_A, uint64_t addr_B, uint64_t addr_C,
                       KPUMatrixDim dim_A, KPUMatrixDim dim_B);

KPUError kpu_matmul_accumulate_f32(KPUHandle handle,
                                  uint64_t addr_A, uint64_t addr_B, uint64_t addr_C,
                                  KPUMatrixDim dim_A, KPUMatrixDim dim_B);

// Utility functions
const char* kpu_error_string(KPUError error);

#ifdef __cplusplus
}
#endif

// C++ Implementation
#ifdef __cplusplus

#include "kpu_simulator.hpp"
#include <exception>
#include <cstring>

namespace {
    KPUError handle_exception() {
        try {
            throw;
        } catch (const std::out_of_range&) {
            return KPU_ERROR_OUT_OF_BOUNDS;
        } catch (const std::invalid_argument&) {
            return KPU_ERROR_INVALID_DIMENSIONS;
        } catch (...) {
            return KPU_ERROR_UNKNOWN;
        }
    }
}

extern "C" {

KPUHandle kpu_create(uint64_t main_memory_size, uint64_t scratchpad_size) {
    try {
        if (main_memory_size == 0) main_memory_size = stillwater::kpu::DEFAULT_MAIN_MEMORY_SIZE;
        if (scratchpad_size == 0) scratchpad_size = stillwater::kpu::DEFAULT_SCRATCHPAD_SIZE;
        
        auto* simulator = new stillwater::kpu::KPUSimulator(main_memory_size, scratchpad_size);
        return reinterpret_cast<KPUHandle>(simulator);
    } catch (...) {
        return nullptr;
    }
}

void kpu_destroy(KPUHandle handle) {
    if (handle) {
        delete reinterpret_cast<stillwater::kpu::KPUSimulator*>(handle);
    }
}

KPUError kpu_main_memory_read(KPUHandle handle, uint64_t addr, void* data, uint64_t size) {
    if (!handle || !data) return KPU_ERROR_NULL_POINTER;
    
    try {
        auto* simulator = reinterpret_cast<stillwater::kpu::KPUSimulator*>(handle);
        simulator->main_memory().read(addr, data, size);
        return KPU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

KPUError kpu_main_memory_write(KPUHandle handle, uint64_t addr, const void* data, uint64_t size) {
    if (!handle || !data) return KPU_ERROR_NULL_POINTER;
    
    try {
        auto* simulator = reinterpret_cast<stillwater::kpu::KPUSimulator*>(handle);
        simulator->main_memory().write(addr, data, size);
        return KPU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

KPUError kpu_scratchpad_read(KPUHandle handle, uint64_t addr, void* data, uint64_t size) {
    if (!handle || !data) return KPU_ERROR_NULL_POINTER;
    
    try {
        auto* simulator = reinterpret_cast<stillwater::kpu::KPUSimulator*>(handle);
        simulator->scratchpad().read(addr, data, size);
        return KPU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

KPUError kpu_scratchpad_write(KPUHandle handle, uint64_t addr, const void* data, uint64_t size) {
    if (!handle || !data) return KPU_ERROR_NULL_POINTER;
    
    try {
        auto* simulator = reinterpret_cast<stillwater::kpu::KPUSimulator*>(handle);
        simulator->scratchpad().write(addr, data, size);
        return KPU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

uint64_t kpu_main_memory_size(KPUHandle handle) {
    if (!handle) return 0;
    auto* simulator = reinterpret_cast<stillwater::kpu::KPUSimulator*>(handle);
    return simulator->main_memory().size();
}

uint64_t kpu_scratchpad_size(KPUHandle handle) {
    if (!handle) return 0;
    auto* simulator = reinterpret_cast<stillwater::kpu::KPUSimulator*>(handle);
    return simulator->scratchpad().size();
}

KPUError kpu_dma_transfer_sync(KPUHandle handle,
                              uint64_t src_addr, uint64_t dst_addr, uint64_t size,
                              bool src_is_main_memory, bool dst_is_main_memory) {
    if (!handle) return KPU_ERROR_NULL_POINTER;
    
    try {
        auto* simulator = reinterpret_cast<stillwater::kpu::KPUSimulator*>(handle);
        simulator->dma_engine().transfer_sync(src_addr, dst_addr, size, 
                                            src_is_main_memory, dst_is_main_memory);
        return KPU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

KPUError kpu_matmul_f32(KPUHandle handle,
                       uint64_t addr_A, uint64_t addr_B, uint64_t addr_C,
                       KPUMatrixDim dim_A, KPUMatrixDim dim_B) {
    if (!handle) return KPU_ERROR_NULL_POINTER;
    
    try {
        auto* simulator = reinterpret_cast<stillwater::kpu::KPUSimulator*>(handle);
        stillwater::kpu::MatrixDim cpp_dim_A{dim_A.rows, dim_A.cols};
        stillwater::kpu::MatrixDim cpp_dim_B{dim_B.rows, dim_B.cols};
        
        simulator->compute_engine().matmul_f32(addr_A, addr_B, addr_C, cpp_dim_A, cpp_dim_B);
        return KPU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

KPUError kpu_matmul_accumulate_f32(KPUHandle handle,
                                  uint64_t addr_A, uint64_t addr_B, uint64_t addr_C,
                                  KPUMatrixDim dim_A, KPUMatrixDim dim_B) {
    if (!handle) return KPU_ERROR_NULL_POINTER;
    
    try {
        auto* simulator = reinterpret_cast<stillwater::kpu::KPUSimulator*>(handle);
        stillwater::kpu::MatrixDim cpp_dim_A{dim_A.rows, dim_A.cols};
        stillwater::kpu::MatrixDim cpp_dim_B{dim_B.rows, dim_B.cols};
        
        simulator->compute_engine().matmul_accumulate_f32(addr_A, addr_B, addr_C, cpp_dim_A, cpp_dim_B);
        return KPU_SUCCESS;
    } catch (...) {
        return handle_exception();
    }
}

const char* kpu_error_string(KPUError error) {
    switch (error) {
        case KPU_SUCCESS: return "Success";
        case KPU_ERROR_INVALID_HANDLE: return "Invalid handle";
        case KPU_ERROR_OUT_OF_BOUNDS: return "Memory access out of bounds";
        case KPU_ERROR_INVALID_DIMENSIONS: return "Invalid matrix dimensions";
        case KPU_ERROR_NULL_POINTER: return "Null pointer";
        case KPU_ERROR_UNKNOWN: return "Unknown error";
        default: return "Invalid error code";
    }
}

} // extern "C"

#endif // __cplusplus