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

namespace sw::kpu {

// External memory component - manages its own memory model
class KPU_API ExternalMemory {
private:
    std::vector<std::uint8_t> memory_model;  // Dynamically managed resource
    Size capacity;
    Size bandwidth_bytes_per_cycle;
    mutable Cycle last_access_cycle;
    
public:
    explicit ExternalMemory(Size capacity_mb = 1024, Size bandwidth_gbps = 100);
    ~ExternalMemory() = default;
    
    // Core memory operations
    void read(Address addr, void* data, Size size);
    void write(Address addr, const void* data, Size size);
    bool is_ready() const;
    
    // Configuration and status
    Size get_capacity() const { return capacity; }
    Size get_bandwidth() const { return bandwidth_bytes_per_cycle; }
    void reset();
    
    // Statistics
    Cycle get_last_access_cycle() const { return last_access_cycle; }
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
