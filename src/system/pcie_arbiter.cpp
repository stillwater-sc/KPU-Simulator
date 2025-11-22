#include <sw/system/pcie_arbiter.hpp>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace sw::system {

PCIeArbiter::PCIeArbiter(double clock_freq_ghz,
                         double link_bandwidth_gb_s,
                         uint32_t max_outstanding_tags)
    : clock_freq_ghz_(clock_freq_ghz)
    , link_bandwidth_gb_s_(link_bandwidth_gb_s)
    , max_outstanding_tags_(max_outstanding_tags)
    , current_cycle_(0)
    , next_tag_(0)
    , tracing_enabled_(false)
    , trace_logger_(&trace::TraceLogger::instance())
{
}

uint32_t PCIeArbiter::enqueue_request(TransactionRequest request) {
    request.arrival_cycle = current_cycle_;

    // Assign tag for non-posted transactions
    if (request.type != TransactionType::MEMORY_WRITE) {
        request.tag = allocate_tag();
    }

    // Route to appropriate queue based on transaction type
    switch (request.type) {
        case TransactionType::CONFIG_WRITE:
        case TransactionType::CONFIG_READ:
        case TransactionType::MEMORY_READ:
            // Non-posted transactions go to command queue
            command_queue_.push(request);
            break;

        case TransactionType::MEMORY_WRITE:
            // Posted transactions go to data queue
            data_queue_.push(request);
            break;

        case TransactionType::COMPLETION:
            // Completions go to completion queue
            completion_queue_.push(request);
            break;
    }

    return request.tag;
}

void PCIeArbiter::step() {
    current_cycle_++;

    // PCIe bus is a shared physical resource - only ONE transaction can be active at a time
    // Priority: completion > command > data (completions are typically prioritized)

    // First, check for completed transactions and free their slots
    if (command_slot_.busy && current_cycle_ >= command_slot_.completion_cycle) {
        auto& request = command_slot_.current_request;
        log_transaction_complete(command_slot_, "PCIE_CMD");

        if (request.type == TransactionType::CONFIG_READ ||
            request.type == TransactionType::MEMORY_READ) {
            generate_completion(request);
        }

        if (request.completion_callback) {
            request.completion_callback();
        }

        command_slot_.busy = false;
    }

    if (data_slot_.busy && current_cycle_ >= data_slot_.completion_cycle) {
        auto& request = data_slot_.current_request;
        log_transaction_complete(data_slot_, "PCIE_DATA");

        if (request.completion_callback) {
            request.completion_callback();
        }

        data_slot_.busy = false;
    }

    if (completion_slot_.busy && current_cycle_ >= completion_slot_.completion_cycle) {
        auto& request = completion_slot_.current_request;
        log_transaction_complete(completion_slot_, "PCIE_CPL");
        outstanding_requests_.erase(request.tag);

        if (request.completion_callback) {
            request.completion_callback();
        }

        completion_slot_.busy = false;
    }

    // Only start a new transaction if the bus is completely free
    bool bus_busy = command_slot_.busy || data_slot_.busy || completion_slot_.busy;

    if (!bus_busy) {
        // Arbitrate between queues - priority: completion > command > data
        if (!completion_queue_.empty()) {
            completion_slot_.busy = true;
            completion_slot_.current_request = completion_queue_.front();
            completion_queue_.pop();

            auto& request = completion_slot_.current_request;
            completion_slot_.start_cycle = current_cycle_;
            trace::CycleCount duration = calculate_duration(request);
            completion_slot_.completion_cycle = current_cycle_ + duration;

            log_transaction_start(request, completion_slot_, "PCIE_CPL");
        } else if (!command_queue_.empty()) {
            command_slot_.busy = true;
            command_slot_.current_request = command_queue_.front();
            command_queue_.pop();

            auto& request = command_slot_.current_request;
            command_slot_.start_cycle = current_cycle_;
            trace::CycleCount duration = calculate_duration(request);
            command_slot_.completion_cycle = current_cycle_ + duration;

            if (request.type != TransactionType::CONFIG_WRITE) {
                outstanding_requests_[request.tag] = request;
            }

            log_transaction_start(request, command_slot_, "PCIE_CMD");
        } else if (!data_queue_.empty()) {
            data_slot_.busy = true;
            data_slot_.current_request = data_queue_.front();
            data_queue_.pop();

            auto& request = data_slot_.current_request;
            data_slot_.start_cycle = current_cycle_;
            trace::CycleCount duration = calculate_duration(request);
            data_slot_.completion_cycle = current_cycle_ + duration;

            log_transaction_start(request, data_slot_, "PCIE_DATA");
        }
    }
}

bool PCIeArbiter::is_busy() const {
    return command_slot_.busy || data_slot_.busy || completion_slot_.busy ||
           !command_queue_.empty() || !data_queue_.empty() || !completion_queue_.empty();
}

void PCIeArbiter::enable_tracing(bool enabled, trace::TraceLogger* logger) {
    tracing_enabled_ = enabled;
    if (logger) {
        trace_logger_ = logger;
    }
}

void PCIeArbiter::reset() {
    // Clear all queues
    command_queue_ = {};
    data_queue_ = {};
    completion_queue_ = {};

    // Reset slots
    command_slot_.busy = false;
    data_slot_.busy = false;
    completion_slot_.busy = false;

    // Clear outstanding requests
    outstanding_requests_.clear();

    current_cycle_ = 0;
    next_tag_ = 0;
}

void PCIeArbiter::process_command_queue() {
    // Check if current command transaction completed
    if (command_slot_.busy && current_cycle_ >= command_slot_.completion_cycle) {
        auto& request = command_slot_.current_request;

        // Log completion trace
        log_transaction_complete(command_slot_, "PCIE_CMD");

        // Generate completion for non-posted transaction
        if (request.type == TransactionType::CONFIG_READ ||
            request.type == TransactionType::MEMORY_READ) {
            generate_completion(request);
        }

        // Call completion callback
        if (request.completion_callback) {
            request.completion_callback();
        }

        // Free the slot
        command_slot_.busy = false;
    }

    // Start new command transaction if slot is free
    if (!command_slot_.busy && !command_queue_.empty()) {
        command_slot_.busy = true;
        command_slot_.current_request = command_queue_.front();
        command_queue_.pop();

        auto& request = command_slot_.current_request;

        // Record start time and calculate completion time
        command_slot_.start_cycle = current_cycle_;
        trace::CycleCount duration = calculate_duration(request);
        command_slot_.completion_cycle = current_cycle_ + duration;

        // Track non-posted request for completion
        if (request.type != TransactionType::CONFIG_WRITE) {
            outstanding_requests_[request.tag] = request;
        }

        // Log transaction start
        log_transaction_start(request, command_slot_, "PCIE_CMD");
    }
}

void PCIeArbiter::process_data_queue() {
    // Check if current data transaction completed
    if (data_slot_.busy && current_cycle_ >= data_slot_.completion_cycle) {
        auto& request = data_slot_.current_request;

        // Log completion trace
        log_transaction_complete(data_slot_, "PCIE_DATA");

        // Call completion callback (even though it's posted, we may need notification)
        if (request.completion_callback) {
            request.completion_callback();
        }

        // Free the slot
        data_slot_.busy = false;
    }

    // Start new data transaction if slot is free
    if (!data_slot_.busy && !data_queue_.empty()) {
        data_slot_.busy = true;
        data_slot_.current_request = data_queue_.front();
        data_queue_.pop();

        auto& request = data_slot_.current_request;

        // Record start time and calculate completion time
        data_slot_.start_cycle = current_cycle_;
        trace::CycleCount duration = calculate_duration(request);
        data_slot_.completion_cycle = current_cycle_ + duration;

        // Log transaction start
        log_transaction_start(request, data_slot_, "PCIE_DATA");
    }
}

void PCIeArbiter::process_completion_queue() {
    // Check if current completion transaction completed
    if (completion_slot_.busy && current_cycle_ >= completion_slot_.completion_cycle) {
        auto& request = completion_slot_.current_request;

        // Log completion trace
        log_transaction_complete(completion_slot_, "PCIE_CPL");

        // Remove from outstanding requests
        outstanding_requests_.erase(request.tag);

        // Call completion callback
        if (request.completion_callback) {
            request.completion_callback();
        }

        // Free the slot
        completion_slot_.busy = false;
    }

    // Start new completion transaction if slot is free
    if (!completion_slot_.busy && !completion_queue_.empty()) {
        completion_slot_.busy = true;
        completion_slot_.current_request = completion_queue_.front();
        completion_queue_.pop();

        auto& request = completion_slot_.current_request;

        // Record start time and calculate completion time
        completion_slot_.start_cycle = current_cycle_;
        trace::CycleCount duration = calculate_duration(request);
        completion_slot_.completion_cycle = current_cycle_ + duration;

        // Log transaction start
        log_transaction_start(request, completion_slot_, "PCIE_CPL");
    }
}

void PCIeArbiter::generate_completion(const TransactionRequest& request) {
    // Create completion TLP for the original request
    TransactionRequest completion;
    completion.type = TransactionType::COMPLETION;
    completion.arrival_cycle = current_cycle_;
    completion.transfer_size = request.transfer_size;
    completion.tag = request.tag;
    completion.requester_id = request.requester_id;
    completion.description = "Completion for: " + request.description;
    completion.completion_callback = nullptr;  // Completion of completion handled by original request

    // Swap source/destination for return path
    completion.src_addr = request.dst_addr;
    completion.dst_addr = request.src_addr;
    completion.src_component = request.dst_component;
    completion.dst_component = request.src_component;
    completion.src_id = request.dst_id;
    completion.dst_id = request.src_id;

    // Calculate duration for completion (typically same size as request for reads)
    completion.duration_cycles = calculate_duration(completion);

    completion_queue_.push(completion);
}

trace::CycleCount PCIeArbiter::calculate_duration(const TransactionRequest& request) const {
    // All transaction types use the same PCIe link bandwidth
    // The duration depends only on transfer size, not transaction type

    // Calculate cycles based on transfer size and bandwidth
    // link_bandwidth_gb_s is in GB/s, transfer_size is in bytes
    // bytes_per_cycle = (link_bandwidth_gb_s * 1e9) / (clock_freq_ghz * 1e9)
    //                 = link_bandwidth_gb_s / clock_freq_ghz
    double bytes_per_cycle = link_bandwidth_gb_s_ / clock_freq_ghz_;
    trace::CycleCount cycles = static_cast<trace::CycleCount>(
        std::ceil(static_cast<double>(request.transfer_size) / bytes_per_cycle)
    );

    // Minimum 1 cycle for any transaction
    return std::max<trace::CycleCount>(1, cycles);
}

void PCIeArbiter::log_transaction_start(const TransactionRequest& /* request */,
                                        TransactionSlot& slot,
                                        const std::string& /* queue_name */) {
    if (!tracing_enabled_ || !trace_logger_) {
        return;
    }

    // Just allocate transaction ID for now, actual logging happens at completion
    slot.trace_txn_id = trace_logger_->next_transaction_id();
}

void PCIeArbiter::log_transaction_complete(const TransactionSlot& slot,
                                          const std::string& queue_name) {
    if (!tracing_enabled_ || !trace_logger_) {
        return;
    }

    const auto& request = slot.current_request;

    // Log memory operations for MEMORY_READ and MEMORY_WRITE transactions
    // This shows the source and destination resource occupancy
    if (request.type == TransactionType::MEMORY_READ || request.type == TransactionType::MEMORY_WRITE) {
        // Source READ event (e.g., HOST_MEMORY read or KPU_MEMORY read)
        trace::TraceEntry read_entry(
            slot.start_cycle,
            request.src_component,
            request.src_id,
            trace::TransactionType::READ,
            slot.trace_txn_id
        );
        read_entry.clock_freq_ghz = clock_freq_ghz_;
        trace::CycleCount read_latency = 1;  // Minimum 1 cycle for memory access
        read_entry.complete(slot.start_cycle + read_latency, trace::TransactionStatus::COMPLETED);

        trace::MemoryPayload read_payload;
        read_payload.location = trace::MemoryLocation(
            request.src_addr, request.transfer_size,
            request.src_id, request.src_component
        );
        read_payload.is_hit = true;
        read_payload.latency_cycles = static_cast<uint32_t>(read_latency);
        read_entry.payload = read_payload;
        read_entry.description = "PCIe source read";
        trace_logger_->log(std::move(read_entry));

        // Destination WRITE event (e.g., KPU_MEMORY write or HOST_MEMORY write)
        trace::TraceEntry write_entry(
            slot.completion_cycle - 1,  // Write happens at end of transfer
            request.dst_component,
            request.dst_id,
            trace::TransactionType::WRITE,
            slot.trace_txn_id
        );
        write_entry.clock_freq_ghz = clock_freq_ghz_;
        trace::CycleCount write_latency = 1;  // Minimum 1 cycle for memory access
        write_entry.complete(slot.completion_cycle, trace::TransactionStatus::COMPLETED);

        trace::MemoryPayload write_payload;
        write_payload.location = trace::MemoryLocation(
            request.dst_addr, request.transfer_size,
            request.dst_id, request.dst_component
        );
        write_payload.is_hit = true;
        write_payload.latency_cycles = static_cast<uint32_t>(write_latency);
        write_entry.payload = write_payload;
        write_entry.description = "PCIe destination write";
        trace_logger_->log(std::move(write_entry));
    }

    // Log PCIe bus transfer event
    trace::ComponentType component_type = trace::ComponentType::PCIE_BUS;

    trace::TraceEntry entry(
        slot.start_cycle,
        component_type,
        0,
        trace::TransactionType::TRANSFER,
        slot.trace_txn_id
    );

    entry.clock_freq_ghz = clock_freq_ghz_;
    entry.complete(slot.completion_cycle, trace::TransactionStatus::COMPLETED);

    // Create DMA payload
    trace::DMAPayload payload;
    payload.source = trace::MemoryLocation(
        request.src_addr, request.transfer_size,
        request.src_id, request.src_component
    );
    payload.destination = trace::MemoryLocation(
        request.dst_addr, request.transfer_size,
        request.dst_id, request.dst_component
    );
    payload.bytes_transferred = request.transfer_size;
    payload.bandwidth_gb_s = link_bandwidth_gb_s_;  // All transactions use the same link bandwidth

    entry.payload = payload;

    // Build description with queue and transaction type information
    std::ostringstream desc;
    desc << "[" << queue_name << "] ";
    switch (request.type) {
        case TransactionType::CONFIG_WRITE:
            desc << "Config Write: ";
            break;
        case TransactionType::CONFIG_READ:
            desc << "Config Read: ";
            break;
        case TransactionType::MEMORY_READ:
            desc << "Memory Read: ";
            break;
        case TransactionType::MEMORY_WRITE:
            desc << "Memory Write: ";
            break;
        case TransactionType::COMPLETION:
            desc << "Completion: ";
            break;
    }
    desc << request.description;

    if (request.tag != 0) {
        desc << " (tag:" << request.tag << ")";
    }

    entry.description = desc.str();

    trace_logger_->log(std::move(entry));
}

uint32_t PCIeArbiter::allocate_tag() {
    // Simple tag allocation (wrap around)
    // In real hardware, would need to check for available tags
    uint32_t tag = next_tag_++;
    if (next_tag_ >= max_outstanding_tags_) {
        next_tag_ = 0;
    }
    return tag;
}

} // namespace sw::system
