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
#include <sw/kpu/components/scratchpad.hpp>

namespace sw::kpu {
    
// DMA Engine for data movement between memory hierarchies
class KPU_API DMAEngine {
public:
    enum class MemoryType {
        EXTERNAL,
        SCRATCHPAD
    };

    struct Transfer {
        MemoryType src_type;
        size_t src_id;
        Address src_addr;
        MemoryType dst_type;
        size_t dst_id;
        Address dst_addr;
        Size size;
        std::function<void()> completion_callback;
    };

private:
    std::vector<Transfer> transfer_queue;  // Dynamically managed resource
    bool is_active;
    size_t engine_id;  // For debugging/identification

public:
    explicit DMAEngine(size_t engine_id = 0);
    ~DMAEngine() = default;

    // Transfer operations - now configured per-transfer
    void enqueue_transfer(MemoryType src_type, size_t src_id, Address src_addr,
                         MemoryType dst_type, size_t dst_id, Address dst_addr,
                         Size size, std::function<void()> callback = nullptr);
    bool process_transfers(std::vector<ExternalMemory>& memory_banks,
                          std::vector<Scratchpad>& scratchpads);
    bool is_busy() const { return is_active || !transfer_queue.empty(); }
    void reset();

    // Status and identification
    size_t get_engine_id() const { return engine_id; }
    size_t get_queue_size() const { return transfer_queue.size(); }
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif