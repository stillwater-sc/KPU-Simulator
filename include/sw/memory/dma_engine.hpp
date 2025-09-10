#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>

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
#include <sw/memory/external_memory.hpp>
#include <sw/memory/scratchpad.hpp>

namespace sw::kpu {
    
// DMA Engine for data movement between memory hierarchies
class KPU_API DMAEngine {
public:
    struct Transfer {
        Address src_addr;
        Address dst_addr;
        Size size;
        std::function<void()> completion_callback;
    };
    
    enum class MemoryType {
        EXTERNAL,
        SCRATCHPAD
    };
    
private:
    std::vector<Transfer> transfer_queue;  // Dynamically managed resource
    bool is_active;
    MemoryType src_type;
    MemoryType dst_type;
    size_t src_id;  // Index into memory bank vector
    size_t dst_id;  // Index into memory bank vector
    
public:
    DMAEngine(MemoryType src_type, size_t src_id, MemoryType dst_type, size_t dst_id);
    ~DMAEngine() = default;
    
    // Transfer operations
    void enqueue_transfer(Address src_addr, Address dst_addr, Size size, 
                         std::function<void()> callback = nullptr);
    bool process_transfers(std::vector<ExternalMemory>& memory_banks, 
                          std::vector<Scratchpad>& scratchpads);
    bool is_busy() const { return is_active || !transfer_queue.empty(); }
    void reset();
    
    // Configuration
    MemoryType get_src_type() const { return src_type; }
    MemoryType get_dst_type() const { return dst_type; }
    size_t get_src_id() const { return src_id; }
    size_t get_dst_id() const { return dst_id; }
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif