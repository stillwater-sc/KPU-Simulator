#include <vector>
#include <stdexcept>
#include <string>
#include <cstdint>

#include <sw/memory/external_memory.hpp>
#include <sw/kpu/components/dma_engine.hpp>
#include <sw/kpu/components/scratchpad.hpp>

namespace sw::kpu {

// DMAEngine implementation - manages its own transfer queue
DMAEngine::DMAEngine(size_t engine_id)
    : is_active(false), engine_id(engine_id) {
}

void DMAEngine::enqueue_transfer(MemoryType src_type, size_t src_id, Address src_addr,
                                MemoryType dst_type, size_t dst_id, Address dst_addr,
                                Size size, std::function<void()> callback) {
    transfer_queue.emplace_back(Transfer{
        src_type, src_id, src_addr,
        dst_type, dst_id, dst_addr,
        size, std::move(callback)
    });
}

bool DMAEngine::process_transfers(std::vector<ExternalMemory>& memory_banks,
                                 std::vector<Scratchpad>& scratchpads) {
    if (transfer_queue.empty()) {
        is_active = false;
        return false;
    }

    is_active = true;
    auto& transfer = transfer_queue.front();

    // Allocate temporary buffer for the transfer
    std::vector<std::uint8_t> buffer(transfer.size);

    // Read from source
    if (transfer.src_type == MemoryType::EXTERNAL) {
        if (transfer.src_id >= memory_banks.size()) {
            throw std::out_of_range("Invalid source memory bank ID: " + std::to_string(transfer.src_id));
        }
        memory_banks[transfer.src_id].read(transfer.src_addr, buffer.data(), transfer.size);
    } else {
        if (transfer.src_id >= scratchpads.size()) {
            throw std::out_of_range("Invalid source scratchpad ID: " + std::to_string(transfer.src_id));
        }
        scratchpads[transfer.src_id].read(transfer.src_addr, buffer.data(), transfer.size);
    }

    // Write to destination
    if (transfer.dst_type == MemoryType::EXTERNAL) {
        if (transfer.dst_id >= memory_banks.size()) {
            throw std::out_of_range("Invalid destination memory bank ID: " + std::to_string(transfer.dst_id));
        }
        memory_banks[transfer.dst_id].write(transfer.dst_addr, buffer.data(), transfer.size);
    } else {
        if (transfer.dst_id >= scratchpads.size()) {
            throw std::out_of_range("Invalid destination scratchpad ID: " + std::to_string(transfer.dst_id));
        }
        scratchpads[transfer.dst_id].write(transfer.dst_addr, buffer.data(), transfer.size);
    }

    // Call completion callback if provided
    if (transfer.completion_callback) {
        transfer.completion_callback();
    }

    transfer_queue.erase(transfer_queue.begin());

    bool completed = transfer_queue.empty();
    if (completed) {
        is_active = false;
    }

    return completed;
}

void DMAEngine::reset() {
    transfer_queue.clear();
    is_active = false;
}

} // namespace sw::kpu