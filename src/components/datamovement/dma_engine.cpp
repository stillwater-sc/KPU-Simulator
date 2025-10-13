#include <vector>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <cmath>

#include <sw/memory/external_memory.hpp>
#include <sw/kpu/components/dma_engine.hpp>
#include <sw/kpu/components/scratchpad.hpp>
#include <sw/kpu/components/l3_tile.hpp>
#include <sw/kpu/components/l2_bank.hpp>

namespace sw::kpu {

// DMAEngine implementation - cycle-accurate multi-cycle processing like BlockMover
DMAEngine::DMAEngine(size_t engine_id, double clock_freq_ghz, double bandwidth_gb_s)
    : is_active(false)
    , engine_id(engine_id)
    , cycles_remaining(0)
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
        0,               // start_cycle (will be set when transfer actually starts)
        0,               // end_cycle (not yet completed)
        txn_id
    };

    transfer_queue.emplace_back(std::move(transfer));
}

bool DMAEngine::process_transfers(std::vector<ExternalMemory>& memory_banks,
                                  std::vector<L3Tile>& l3_tiles,
                                  std::vector<L2Bank>& l2_banks,
                                  std::vector<Scratchpad>& scratchpads) {
    if (transfer_queue.empty() && cycles_remaining == 0) {
        is_active = false;
        return false;
    }

    is_active = true;

    // Start a new transfer if none is active
    if (cycles_remaining == 0 && !transfer_queue.empty()) {
        auto& transfer = transfer_queue.front();

        // Set the actual start cycle now that processing begins
        transfer.start_cycle = current_cycle_;

        // Validate destination capacity before starting the transfer
        if (transfer.dst_type == MemoryType::SCRATCHPAD) {
            if (transfer.dst_id >= scratchpads.size()) {
                throw std::out_of_range("Invalid destination scratchpad ID: " + std::to_string(transfer.dst_id));
            }
            if (transfer.dst_addr + transfer.size > scratchpads[transfer.dst_id].get_capacity()) {
                throw std::out_of_range("DMA transfer would exceed scratchpad capacity: addr=" +
                    std::to_string(transfer.dst_addr) + " size=" + std::to_string(transfer.size) +
                    " capacity=" + std::to_string(scratchpads[transfer.dst_id].get_capacity()));
            }
        }

        // Calculate transfer latency in cycles based on bandwidth
        // bandwidth_gb_s_ is in GB/s, size is in bytes
        // bytes_per_cycle = (bandwidth_gb_s * 1e9) / (clock_freq_ghz * 1e9)
        //                 = bandwidth_gb_s / clock_freq_ghz
        double bytes_per_cycle = bandwidth_gb_s_ / clock_freq_ghz_;
        cycles_remaining = static_cast<trace::CycleCount>(std::ceil(transfer.size / bytes_per_cycle));
        if (cycles_remaining == 0) cycles_remaining = 1;  // Minimum 1 cycle

        // Allocate buffer for the transfer
        transfer_buffer.resize(transfer.size);

        // Log trace entry for transfer issue (now that it's actually starting)
        if (tracing_enabled_ && trace_logger_) {
            trace::TraceEntry entry(
                current_cycle_,
                trace::ComponentType::DMA_ENGINE,
                static_cast<uint32_t>(engine_id),
                trace::TransactionType::TRANSFER,
                transfer.transaction_id
            );

            // Set clock frequency for time conversion
            entry.clock_freq_ghz = clock_freq_ghz_;

            // Map MemoryType to ComponentType
            auto to_component_type = [](MemoryType type) {
                switch (type) {
                    case MemoryType::HOST_MEMORY: return trace::ComponentType::HOST_MEMORY;
                    case MemoryType::EXTERNAL: return trace::ComponentType::EXTERNAL_MEMORY;
                    case MemoryType::L3_TILE: return trace::ComponentType::L3_TILE;
                    case MemoryType::L2_BANK: return trace::ComponentType::L2_BANK;
                    case MemoryType::SCRATCHPAD: return trace::ComponentType::SCRATCHPAD;
                    default: return trace::ComponentType::UNKNOWN;
                }
            };

            // Create DMA payload
            trace::DMAPayload payload;
            payload.source = trace::MemoryLocation(
                transfer.src_addr, transfer.size, static_cast<uint32_t>(transfer.src_id),
                to_component_type(transfer.src_type)
            );
            payload.destination = trace::MemoryLocation(
                transfer.dst_addr, transfer.size, static_cast<uint32_t>(transfer.dst_id),
                to_component_type(transfer.dst_type)
            );
            payload.bytes_transferred = transfer.size;
            payload.bandwidth_gb_s = bandwidth_gb_s_;

            entry.payload = payload;
            entry.description = "DMA transfer issued";

            trace_logger_->log(std::move(entry));
        }

        // Read from source into buffer
        switch (transfer.src_type) {
            case MemoryType::HOST_MEMORY:
                // Host memory is external - for simulation, this is a no-op
                // The data should already be in the destination from functional model
                break;

            case MemoryType::EXTERNAL:
                if (transfer.src_id >= memory_banks.size()) {
                    throw std::out_of_range("Invalid source memory bank ID: " + std::to_string(transfer.src_id));
                }
                memory_banks[transfer.src_id].read(transfer.src_addr, transfer_buffer.data(), transfer.size);
                break;

            case MemoryType::L3_TILE:
                if (transfer.src_id >= l3_tiles.size()) {
                    throw std::out_of_range("Invalid source L3 tile ID: " + std::to_string(transfer.src_id));
                }
                l3_tiles[transfer.src_id].read(transfer.src_addr, transfer_buffer.data(), transfer.size);
                break;

            case MemoryType::L2_BANK:
                if (transfer.src_id >= l2_banks.size()) {
                    throw std::out_of_range("Invalid source L2 bank ID: " + std::to_string(transfer.src_id));
                }
                l2_banks[transfer.src_id].read(transfer.src_addr, transfer_buffer.data(), transfer.size);
                break;

            case MemoryType::SCRATCHPAD:
                if (transfer.src_id >= scratchpads.size()) {
                    throw std::out_of_range("Invalid source scratchpad ID: " + std::to_string(transfer.src_id));
                }
                scratchpads[transfer.src_id].read(transfer.src_addr, transfer_buffer.data(), transfer.size);
                break;
        }
    }

    // Process one cycle of the current transfer
    if (cycles_remaining > 0) {
        cycles_remaining--;

        // Transfer completes when cycles reach 0
        if (cycles_remaining == 0) {
            auto& transfer = transfer_queue.front();

            // Set completion time
            transfer.end_cycle = current_cycle_;

            // Write to destination
            switch (transfer.dst_type) {
                case MemoryType::HOST_MEMORY:
                    // Host memory is external - for simulation, this is a no-op
                    break;

                case MemoryType::EXTERNAL:
                    if (transfer.dst_id >= memory_banks.size()) {
                        throw std::out_of_range("Invalid destination memory bank ID: " + std::to_string(transfer.dst_id));
                    }
                    memory_banks[transfer.dst_id].write(transfer.dst_addr, transfer_buffer.data(), transfer.size);
                    break;

                case MemoryType::L3_TILE:
                    if (transfer.dst_id >= l3_tiles.size()) {
                        throw std::out_of_range("Invalid destination L3 tile ID: " + std::to_string(transfer.dst_id));
                    }
                    l3_tiles[transfer.dst_id].write(transfer.dst_addr, transfer_buffer.data(), transfer.size);
                    break;

                case MemoryType::L2_BANK:
                    if (transfer.dst_id >= l2_banks.size()) {
                        throw std::out_of_range("Invalid destination L2 bank ID: " + std::to_string(transfer.dst_id));
                    }
                    l2_banks[transfer.dst_id].write(transfer.dst_addr, transfer_buffer.data(), transfer.size);
                    break;

                case MemoryType::SCRATCHPAD:
                    if (transfer.dst_id >= scratchpads.size()) {
                        throw std::out_of_range("Invalid destination scratchpad ID: " + std::to_string(transfer.dst_id));
                    }
                    // Validate transfer doesn't exceed destination capacity
                    if (transfer.dst_addr + transfer.size > scratchpads[transfer.dst_id].get_capacity()) {
                        throw std::out_of_range("DMA transfer would exceed scratchpad capacity: addr=" +
                            std::to_string(transfer.dst_addr) + " size=" + std::to_string(transfer.size) +
                            " capacity=" + std::to_string(scratchpads[transfer.dst_id].get_capacity()));
                    }
                    scratchpads[transfer.dst_id].write(transfer.dst_addr, transfer_buffer.data(), transfer.size);
                    break;
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

                // Map MemoryType to ComponentType
                auto to_component_type = [](MemoryType type) {
                    switch (type) {
                        case MemoryType::HOST_MEMORY: return trace::ComponentType::HOST_MEMORY;
                        case MemoryType::EXTERNAL: return trace::ComponentType::EXTERNAL_MEMORY;
                        case MemoryType::L3_TILE: return trace::ComponentType::L3_TILE;
                        case MemoryType::L2_BANK: return trace::ComponentType::L2_BANK;
                        case MemoryType::SCRATCHPAD: return trace::ComponentType::SCRATCHPAD;
                        default: return trace::ComponentType::UNKNOWN;
                    }
                };

                // Create DMA payload
                trace::DMAPayload payload;
                payload.source = trace::MemoryLocation(
                    transfer.src_addr, transfer.size, static_cast<uint32_t>(transfer.src_id),
                    to_component_type(transfer.src_type)
                );
                payload.destination = trace::MemoryLocation(
                    transfer.dst_addr, transfer.size, static_cast<uint32_t>(transfer.dst_id),
                    to_component_type(transfer.dst_type)
                );
                payload.bytes_transferred = transfer.size;
                payload.bandwidth_gb_s = bandwidth_gb_s_;

                entry.payload = payload;
                entry.description = "DMA transfer completed";

                trace_logger_->log(std::move(entry));
            }

            // Call completion callback if provided
            if (transfer.completion_callback) {
                transfer.completion_callback();
            }

            // Remove completed transfer from queue
            transfer_queue.erase(transfer_queue.begin());
            transfer_buffer.clear();

            // Check if all work is done
            bool completed = transfer_queue.empty();
            if (completed) {
                is_active = false;
            }

            return completed;
        }
    }

    return false;
}

void DMAEngine::reset() {
    transfer_queue.clear();
    transfer_buffer.clear();
    cycles_remaining = 0;
    is_active = false;
    current_cycle_ = 0;
}

} // namespace sw::kpu
