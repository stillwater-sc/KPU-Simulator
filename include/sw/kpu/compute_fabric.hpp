#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <stdexcept>
#include <iomanip>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251) // DLL interface warnings
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

#include <sw/concepts.hpp>
#include <sw/kpu/scratchpad.hpp>

namespace sw::kpu {

// Compute fabric for matrix operations
class KPU_API ComputeFabric {
public:
    struct MatMulConfig {
        Size m, n, k; // Matrix dimensions: C[m,n] = A[m,k] * B[k,n]
        Address a_addr, b_addr, c_addr; // Addresses in scratchpad
        size_t scratchpad_id; // Which scratchpad to use
        std::function<void()> completion_callback;
    };
    
private:
    bool is_computing;
    Cycle compute_start_cycle;
    MatMulConfig current_op;
    size_t tile_id;  // Which compute tile this fabric represents
    
public:
    explicit ComputeFabric(size_t tile_id);
    ~ComputeFabric() = default;
    
    // Compute operations
    void start_matmul(const MatMulConfig& config);
    bool update(Cycle current_cycle, std::vector<Scratchpad>& scratchpads);
    bool is_busy() const { return is_computing; }
    void reset();
    
    // Configuration
    size_t get_tile_id() const { return tile_id; }
    
private:
    void execute_matmul(std::vector<Scratchpad>& scratchpads);
    Cycle estimate_cycles(Size m, Size n, Size k) const;
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif