#include <vector>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <cmath>

#include <sw/memory/external_memory.hpp>
#include <sw/kpu/components/dma_engine.hpp>
#include <sw/kpu/components/scratchpad.hpp>

namespace sw::kpu {

// DMAEngine implementation - manages its own transfer queue with cycle-based timing
DMAEngine::DMAEngine(size_t engine_id, double clock_freq_ghz, double bandwidth_gb_s)
    : is_active(false)
    , engine_id(engine_id)
    , tracing_enabled_(false)
    , trace_logger_(&trace::TraceLogger::instance())
    , clock_freq_ghz_(clock_freq_ghz)
    , bandwidth_gb_s_(bandwidth_gb_s)
    , current_cycle_(0)
{
}

void DMAEngine::enqueue_transfer(MemoryType src_type, size_t src_id, Address src_addr,
                                MemoryType dst_type, size_t dst_id, Address dst_addr,
                                Size size, std::function<void()> callback) {
    // Get transaction ID
    uint64_t txn_id = trace_logger_->next_transaction_id();

    // Create transfer with timing info
    Transfer transfer{
        src_type, src_id, src_addr,
        dst_type, dst_id, dst_addr,
        size, std::move(callback),
        current_cycle_,  // start_cycle
        0,               // end_cycle (not yet completed)
        txn_id
    };

    transfer_queue.emplace_back(std::move(transfer));

    // Log trace entry for transfer issue
    if (tracing_enabled_ && trace_logger_) {
        trace::TraceEntry entry(
            current_cycle_,
            trace::ComponentType::DMA_ENGINE,
            static_cast<uint32_t>(engine_id),
            trace::TransactionType::TRANSFER,
            txn_id
        );

        // Set clock frequency for time conversion
        entry.clock_freq_ghz = clock_freq_ghz_;

        // Create DMA payload
        trace::DMAPayload payload;
        payload.source = trace::MemoryLocation(
            src_addr, size, static_cast<uint32_t>(src_id),
            src_type == MemoryType::EXTERNAL ? trace::ComponentType::EXTERNAL_MEMORY : trace::ComponentType::SCRATCHPAD
        );
        payload.destination = trace::MemoryLocation(
            dst_addr, size, static_cast<uint32_t>(dst_id),
            dst_type == MemoryType::EXTERNAL ? trace::ComponentType::EXTERNAL_MEMORY : trace::ComponentType::SCRATCHPAD
        );
        payload.bytes_transferred = size;
        payload.bandwidth_gb_s = bandwidth_gb_s_;

        entry.payload = payload;
        entry.description = "DMA transfer enqueued";

        trace_logger_->log(std::move(entry));
    }
}

bool DMAEngine::process_transfers(std::vector<ExternalMemory>& memory_banks,
                                 std::vector<Scratchpad>& scratchpads) {
    if (transfer_queue.empty()) {
        is_active = false;
        return false;
    }

    is_active = true;
    auto& transfer = transfer_queue.front();

    // Calculate transfer latency in cycles based on bandwidth
    // bandwidth_gb_s_ is in GB/s, convert to bytes/cycle
    // bytes_per_cycle = (bandwidth_gb_s * 1e9) / (clock_freq_ghz * 1e9)
    //                 = bandwidth_gb_s / clock_freq_ghz
    double bytes_per_cycle = bandwidth_gb_s_ / clock_freq_ghz_;
    uint64_t transfer_cycles = static_cast<uint64_t>(std::ceil(transfer.size / bytes_per_cycle));
    if (transfer_cycles == 0) transfer_cycles = 1;  // Minimum 1 cycle

    // Set completion time
    transfer.end_cycle = current_cycle_ + transfer_cycles;

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

    // Log trace entry for transfer completion
    if (tracing_enabled_ && trace_logger_) {
        trace::TraceEntry entry(
            transfer.start_cycle,
            trace::ComponentType::DMA_ENGINE,
            static_cast<uint32_t>(engine_id),
            trace::TransactionType::TRANSFER,
            transfer.transaction_id
        );

        // Set clock frequency for time conversion
        entry.clock_freq_ghz = clock_freq_ghz_;

        // Complete the entry with end cycle
        entry.complete(transfer.end_cycle, trace::TransactionStatus::COMPLETED);

        // Create DMA payload
        trace::DMAPayload payload;
        payload.source = trace::MemoryLocation(
            transfer.src_addr, transfer.size, static_cast<uint32_t>(transfer.src_id),
            transfer.src_type == MemoryType::EXTERNAL ? trace::ComponentType::EXTERNAL_MEMORY : trace::ComponentType::SCRATCHPAD
        );
        payload.destination = trace::MemoryLocation(
            transfer.dst_addr, transfer.size, static_cast<uint32_t>(transfer.dst_id),
            transfer.dst_type == MemoryType::EXTERNAL ? trace::ComponentType::EXTERNAL_MEMORY : trace::ComponentType::SCRATCHPAD
        );
        payload.bytes_transferred = transfer.size;
        payload.bandwidth_gb_s = bandwidth_gb_s_;

        entry.payload = payload;
        entry.description = "DMA transfer completed";

        trace_logger_->log(std::move(entry));
    }

    // Advance simulation time to transfer completion
    current_cycle_ = transfer.end_cycle;

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
    current_cycle_ = 0;
}

} // namespace sw::kpu