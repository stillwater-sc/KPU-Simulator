#pragma once

#include <queue>
#include <functional>
#include <cstdint>
#include <optional>
#include <sw/concepts.hpp>
#include <sw/trace/trace_logger.hpp>

namespace sw::system {

/**
 * @brief PCIe Arbiter - Models PCIe bus arbitration and transaction serialization
 *
 * This component models the PCIe protocol's separation of command and data phases,
 * implementing proper serialization of transactions on the shared PCIe bus.
 *
 * Architecture:
 * - Command Phase: Non-posted transactions (descriptor writes, config, reads)
 *   - Requires completion packets
 *   - Uses tag-based request/completion matching
 *   - Serialized on command queue
 *
 * - Data Phase: Posted transactions (bulk memory writes)
 *   - Fire-and-forget (no completion required)
 *   - Higher throughput than command phase
 *   - Serialized on data queue
 *
 * Based on PCIe Transaction Layer Packet (TLP) protocol:
 * - Memory Read/Write Requests
 * - Configuration Read/Write
 * - Completion TLPs
 * - Credit-based flow control (modeled as serialization)
 *
 * References:
 * - PCIe Base Specification
 * - https://xillybus.com/tutorials/pci-express-tlp-pcie-primer-tutorial-guide-1
 * - https://www.fpga4fun.com/PCI-Express4.html
 */
class PCIeArbiter {
public:
    /// Transaction types based on PCIe TLP categories
    enum class TransactionType {
        // Non-Posted (require completion)
        CONFIG_WRITE,        ///< Configuration space write (descriptor setup)
        CONFIG_READ,         ///< Configuration space read
        MEMORY_READ,         ///< Memory read request (requires completion)

        // Posted (no completion required)
        MEMORY_WRITE,        ///< Memory write (DMA data transfer)

        // Internal
        COMPLETION           ///< Completion packet for non-posted requests
    };

    /// Transaction request submitted to the arbiter
    struct TransactionRequest {
        TransactionType type;
        trace::CycleCount arrival_cycle;     ///< When request arrived at arbiter
        trace::CycleCount duration_cycles;   ///< Transaction duration (based on size/bandwidth)
        sw::kpu::Size transfer_size;         ///< Size in bytes
        uint32_t requester_id;               ///< Source device ID (e.g., CPU core #)
        uint32_t tag;                        ///< Transaction tag for request/completion matching
        std::string description;             ///< Human-readable description for tracing
        std::function<void()> completion_callback;  ///< Called when transaction completes

        // Source/destination routing information
        sw::kpu::Address src_addr;
        sw::kpu::Address dst_addr;
        trace::ComponentType src_component;
        trace::ComponentType dst_component;
        uint32_t src_id;
        uint32_t dst_id;
    };

    /// Transaction slot state
    struct TransactionSlot {
        bool busy = false;
        TransactionRequest current_request;
        trace::CycleCount start_cycle = 0;   ///< When transaction started processing
        trace::CycleCount completion_cycle = 0;
        uint64_t trace_txn_id = 0;           ///< Trace transaction ID
    };

private:
    // PCIe configuration
    double clock_freq_ghz_;                   ///< Bus clock frequency
    double link_bandwidth_gb_s_;              ///< PCIe link bandwidth (shared by all transaction types)
    uint32_t max_outstanding_tags_;           ///< Maximum outstanding non-posted requests

    // Command queue (non-posted transactions)
    std::queue<TransactionRequest> command_queue_;
    TransactionSlot command_slot_;

    // Data queue (posted transactions - memory writes)
    std::queue<TransactionRequest> data_queue_;
    TransactionSlot data_slot_;

    // Completion tracking for non-posted requests
    std::queue<TransactionRequest> completion_queue_;
    TransactionSlot completion_slot_;
    std::unordered_map<uint32_t, TransactionRequest> outstanding_requests_;  // tag -> request

    // State
    trace::CycleCount current_cycle_;
    uint32_t next_tag_;

    // Tracing
    bool tracing_enabled_;
    trace::TraceLogger* trace_logger_;

public:
    /**
     * @brief Construct PCIe Arbiter
     *
     * @param clock_freq_ghz Bus clock frequency in GHz
     * @param link_bandwidth_gb_s PCIe link bandwidth in GB/s (e.g., 32 GB/s for Gen4 x16)
     * @param max_outstanding_tags Maximum outstanding non-posted requests (typically 32-256)
     */
    PCIeArbiter(double clock_freq_ghz = 1.0,
                double link_bandwidth_gb_s = 32.0,
                uint32_t max_outstanding_tags = 32);

    ~PCIeArbiter() = default;

    // Non-copyable
    PCIeArbiter(const PCIeArbiter&) = delete;
    PCIeArbiter& operator=(const PCIeArbiter&) = delete;

    /**
     * @brief Enqueue a transaction request
     *
     * The arbiter will automatically route the request to the appropriate queue
     * based on transaction type (command vs data).
     *
     * @param request Transaction request to enqueue
     * @return Tag assigned to the request (for non-posted transactions)
     */
    uint32_t enqueue_request(TransactionRequest request);

    /**
     * @brief Step the arbiter by one cycle
     *
     * Processes queued transactions and advances active transactions.
     * Should be called once per simulation cycle.
     */
    void step();

    /**
     * @brief Check if arbiter has any pending or active transactions
     */
    bool is_busy() const;

    /**
     * @brief Get number of pending command transactions
     */
    size_t get_command_queue_depth() const { return command_queue_.size(); }

    /**
     * @brief Get number of pending data transactions
     */
    size_t get_data_queue_depth() const { return data_queue_.size(); }

    /**
     * @brief Get number of outstanding non-posted requests
     */
    size_t get_outstanding_request_count() const { return outstanding_requests_.size(); }

    /**
     * @brief Set current cycle (called by system clock)
     */
    void set_current_cycle(trace::CycleCount cycle) { current_cycle_ = cycle; }

    /**
     * @brief Get current cycle
     */
    trace::CycleCount get_current_cycle() const { return current_cycle_; }

    /**
     * @brief Enable/disable tracing
     */
    void enable_tracing(bool enabled = true, trace::TraceLogger* logger = nullptr);

    /**
     * @brief Reset the arbiter state
     */
    void reset();

private:
    /**
     * @brief Process command queue (non-posted transactions)
     */
    void process_command_queue();

    /**
     * @brief Process data queue (posted transactions)
     */
    void process_data_queue();

    /**
     * @brief Process completion queue
     */
    void process_completion_queue();

    /**
     * @brief Generate completion for non-posted request
     */
    void generate_completion(const TransactionRequest& request);

    /**
     * @brief Calculate transaction duration in cycles
     */
    trace::CycleCount calculate_duration(const TransactionRequest& request) const;

    /**
     * @brief Log trace event for transaction start
     */
    void log_transaction_start(const TransactionRequest& request, TransactionSlot& slot,
                               const std::string& queue_name);

    /**
     * @brief Log trace event for transaction completion
     */
    void log_transaction_complete(const TransactionSlot& slot, const std::string& queue_name);

    /**
     * @brief Allocate next available tag
     */
    uint32_t allocate_tag();
};

} // namespace sw::system
