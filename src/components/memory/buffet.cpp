#include <sw/kpu/components/buffet.hpp>
#include <sw/kpu/components/block_mover.hpp>
#include <sw/kpu/components/streamer.hpp>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sw::kpu {

// Buffet Implementation
Buffet::Buffet(size_t buffet_id, size_t num_banks, const BankConfig& default_config)
    : num_banks(num_banks), buffet_id(buffet_id), next_sequence_id(1) {

    bank_configs.resize(num_banks, default_config);
    bank_states.reserve(num_banks);

    // Initialize bank states
    for (size_t i = 0; i < num_banks; ++i) {
        auto bank_state = std::make_unique<BankState>();
        bank_state->capacity = default_config.bank_size_kb * 1024;
        bank_state->data.resize(bank_state->capacity, 0);
        bank_state->current_occupancy = 0;
        bank_state->is_reading = false;
        bank_state->is_writing = false;
        bank_state->current_phase = EDDOPhase::SYNC;
        bank_state->active_sequence_id = 0;
        bank_state->read_accesses = 0;
        bank_state->write_accesses = 0;
        bank_state->cache_hits = 0;
        bank_state->cache_misses = 0;

        bank_states.push_back(std::move(bank_state));
    }
}

Buffet::Buffet(const Buffet& other)
    : num_banks(other.num_banks)
    , buffet_id(other.buffet_id)
    , bank_configs(other.bank_configs)
    , next_sequence_id(1) {

    // Deep copy bank states (manual copy due to atomic members)
    bank_states.reserve(num_banks);
    for (size_t i = 0; i < num_banks; ++i) {
        auto bank_state = std::make_unique<BankState>();
        const auto& other_bank = *other.bank_states[i];

        bank_state->data = other_bank.data;
        bank_state->capacity = other_bank.capacity;
        bank_state->current_occupancy = other_bank.current_occupancy;
        bank_state->is_reading = other_bank.is_reading.load();
        bank_state->is_writing = other_bank.is_writing.load();
        bank_state->current_phase = other_bank.current_phase;
        bank_state->active_sequence_id = other_bank.active_sequence_id;
        bank_state->read_accesses = other_bank.read_accesses.load();
        bank_state->write_accesses = other_bank.write_accesses.load();
        bank_state->cache_hits = other_bank.cache_hits.load();
        bank_state->cache_misses = other_bank.cache_misses.load();

        bank_states.push_back(std::move(bank_state));
    }

    // Note: We don't copy active commands or dependencies as they're runtime state
}

Buffet& Buffet::operator=(const Buffet& other) {
    if (this != &other) {
        num_banks = other.num_banks;
        buffet_id = other.buffet_id;
        bank_configs = other.bank_configs;

        bank_states.clear();
        bank_states.reserve(num_banks);

        for (size_t i = 0; i < num_banks; ++i) {
            auto bank_state = std::make_unique<BankState>();
            const auto& other_bank = *other.bank_states[i];

            bank_state->data = other_bank.data;
            bank_state->capacity = other_bank.capacity;
            bank_state->current_occupancy = other_bank.current_occupancy;
            bank_state->is_reading = other_bank.is_reading.load();
            bank_state->is_writing = other_bank.is_writing.load();
            bank_state->current_phase = other_bank.current_phase;
            bank_state->active_sequence_id = other_bank.active_sequence_id;
            bank_state->read_accesses = other_bank.read_accesses.load();
            bank_state->write_accesses = other_bank.write_accesses.load();
            bank_state->cache_hits = other_bank.cache_hits.load();
            bank_state->cache_misses = other_bank.cache_misses.load();

            bank_states.push_back(std::move(bank_state));
        }

        // Clear runtime state
        std::lock_guard<std::mutex> cmd_lock(command_mutex);
        command_queue = std::queue<EDDOCommand>();
        active_commands.clear();
        dependency_graph.clear();
    }
    return *this;
}

void Buffet::configure_bank(size_t bank_id, const BankConfig& config) {
    if (bank_id >= num_banks) {
        throw std::out_of_range("Bank ID out of range");
    }

    std::lock_guard<std::mutex> lock(bank_mutex);
    bank_configs[bank_id] = config;

    // Resize bank storage if needed
    Size new_capacity = config.bank_size_kb * 1024;
    if (new_capacity != bank_states[bank_id]->capacity) {
        bank_states[bank_id]->data.resize(new_capacity, 0);
        bank_states[bank_id]->capacity = new_capacity;
        bank_states[bank_id]->current_occupancy = std::min(
            bank_states[bank_id]->current_occupancy, new_capacity);
    }
}

void Buffet::register_block_mover(BlockMover* mover) {
    if (mover != nullptr) {
        block_movers.push_back(mover);
    }
}

void Buffet::register_streamer(Streamer* streamer) {
    if (streamer != nullptr) {
        streamers.push_back(streamer);
    }
}

void Buffet::read(size_t bank_id, Address addr, void* data, Size size) {
    if (!validate_bank_access(bank_id, addr, size)) {
        throw std::out_of_range("Invalid bank read access");
    }

    std::lock_guard<std::mutex> lock(bank_mutex);
    auto& bank = *bank_states[bank_id];

    bank.is_reading = true;

    std::memcpy(data, &bank.data[addr], size);

    bank.read_accesses++;
    bank.is_reading = false;
}

void Buffet::write(size_t bank_id, Address addr, const void* data, Size size) {
    if (!validate_bank_access(bank_id, addr, size)) {
        throw std::out_of_range("Invalid bank write access");
    }

    std::lock_guard<std::mutex> lock(bank_mutex);
    auto& bank = *bank_states[bank_id];

    bank.is_writing = true;

    std::memcpy(&bank.data[addr], data, size);

    // Update occupancy tracking
    Size end_addr = addr + size;
    if (end_addr > bank.current_occupancy) {
        bank.current_occupancy = end_addr;
    }

    bank.write_accesses++;
    bank.is_writing = false;
}

bool Buffet::is_ready(size_t bank_id) const {
    if (bank_id >= num_banks) return false;

    std::lock_guard<std::mutex> lock(bank_mutex);
    const auto& bank = *bank_states[bank_id];
    return !bank.is_reading && !bank.is_writing;
}

void Buffet::enqueue_eddo_command(const EDDOCommand& cmd) {
    std::lock_guard<std::mutex> lock(command_mutex);
    command_queue.push(cmd);
}

bool Buffet::process_eddo_commands() {
    std::lock_guard<std::mutex> lock(command_mutex);

    if (command_queue.empty()) {
        return false;
    }

    // Try to execute ready commands
    bool executed_any = false;
    std::queue<EDDOCommand> deferred_commands;

    while (!command_queue.empty()) {
        EDDOCommand cmd = command_queue.front();
        command_queue.pop();

        if (can_execute_command(cmd)) {
            // Execute based on phase
            switch (cmd.phase) {
                case EDDOPhase::PREFETCH:
                    execute_prefetch_command(cmd);
                    break;
                case EDDOPhase::COMPUTE:
                    execute_compute_command(cmd);
                    break;
                case EDDOPhase::WRITEBACK:
                    execute_writeback_command(cmd);
                    break;
                case EDDOPhase::SYNC:
                    execute_sync_command(cmd);
                    break;
            }
            executed_any = true;
            complete_command(cmd);
        } else {
            // Defer command for later execution
            deferred_commands.push(cmd);
        }
    }

    // Put deferred commands back
    command_queue = std::move(deferred_commands);

    return executed_any;
}

bool Buffet::can_execute_command(const EDDOCommand& cmd) const {
    // Check if bank is available for this phase
    if (!is_bank_available(cmd.bank_id, cmd.phase)) {
        return false;
    }

    // Check dependencies
    for (size_t dep_id : cmd.dependencies) {
        if (active_commands.find(dep_id) != active_commands.end()) {
            return false; // Dependency still active
        }
    }

    return true;
}

void Buffet::execute_prefetch_command(const EDDOCommand& cmd) {
    // Mark bank as transitioning to prefetch phase
    transition_bank_phase(cmd.bank_id, EDDOPhase::PREFETCH, cmd.sequence_id);

    // Simulate prefetch completion immediately for now
    // In a real implementation, this would be asynchronous
    // and complete when the data movement is done

    // For simulation purposes, mark as completed immediately
    // Don't add to active_commands since we're completing immediately
}

void Buffet::execute_compute_command(const EDDOCommand& cmd) {
    transition_bank_phase(cmd.bank_id, EDDOPhase::COMPUTE, cmd.sequence_id);

    // Simulate compute completion immediately for testing
    // In real implementation, this would involve actual computation
    // Don't add to active_commands since we're completing immediately
}

void Buffet::execute_writeback_command(const EDDOCommand& cmd) {
    transition_bank_phase(cmd.bank_id, EDDOPhase::WRITEBACK, cmd.sequence_id);

    // Simulate writeback completion immediately for testing
    // In real implementation, this would involve data streaming
    // Don't add to active_commands since we're completing immediately
}

void Buffet::execute_sync_command(const EDDOCommand& cmd) {
    // Sync commands ensure all previous operations complete
    // Reset bank to sync phase
    transition_bank_phase(cmd.bank_id, EDDOPhase::SYNC, cmd.sequence_id);

    // Sync completes immediately - it's just a coordination point
}

void Buffet::complete_command(const EDDOCommand& cmd) {
    // Remove from active commands
    active_commands.erase(cmd.sequence_id);

    // Update dependencies
    update_dependencies(cmd.sequence_id);

    // Call completion callback if provided
    if (cmd.completion_callback) {
        cmd.completion_callback(cmd);
    }
}

void Buffet::update_dependencies(size_t completed_sequence_id) {
    // Remove completed command from all dependency lists
    for (auto& [seq_id, dependents] : dependency_graph) {
        dependents.erase(
            std::remove(dependents.begin(), dependents.end(), completed_sequence_id),
            dependents.end());
    }
}

bool Buffet::is_bank_available(size_t bank_id, EDDOPhase phase) const {
    if (bank_id >= num_banks) return false;

    const auto& bank = *bank_states[bank_id];

    // More permissive availability check to prevent infinite loops
    switch (phase) {
        case EDDOPhase::PREFETCH:
            return !bank.is_writing;
        case EDDOPhase::COMPUTE:
            // Allow compute on any bank that's not actively reading/writing
            return !bank.is_reading && !bank.is_writing;
        case EDDOPhase::WRITEBACK:
            return !bank.is_reading;
        case EDDOPhase::SYNC:
            // Sync can always proceed - it's a coordination phase
            return true;
    }
    return false;
}

void Buffet::transition_bank_phase(size_t bank_id, EDDOPhase new_phase, size_t sequence_id) {
    if (bank_id >= num_banks) return;

    std::lock_guard<std::mutex> lock(bank_mutex);
    auto& bank = *bank_states[bank_id];
    bank.current_phase = new_phase;
    bank.active_sequence_id = sequence_id;
}

bool Buffet::validate_bank_access(size_t bank_id, Address addr, Size size) const {
    if (bank_id >= num_banks) return false;

    // Check if the access would exceed bank capacity
    return (addr + size) <= bank_states[bank_id]->capacity;
}

Address Buffet::map_to_bank_address(size_t bank_id, Address global_addr) const {
    // Simple mapping - in practice this could be more sophisticated
    return global_addr % bank_states[bank_id]->capacity;
}

void Buffet::orchestrate_double_buffer(size_t bank_a, size_t bank_b,
                                     Address src_addr, Size transfer_size) {
    EDDOCommand prefetch_a{
        .phase = EDDOPhase::PREFETCH,
        .bank_id = bank_a,
        .source_addr = src_addr,
        .dest_addr = 0,
        .transfer_size = transfer_size,
        .sequence_id = next_sequence_id++,
        .dependencies = {},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = nullptr
    };

    EDDOCommand prefetch_b{
        .phase = EDDOPhase::PREFETCH,
        .bank_id = bank_b,
        .source_addr = src_addr + transfer_size,
        .dest_addr = 0,
        .transfer_size = transfer_size,
        .sequence_id = next_sequence_id++,
        .dependencies = {},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = nullptr
    };

    enqueue_eddo_command(prefetch_a);
    enqueue_eddo_command(prefetch_b);
}

void Buffet::orchestrate_pipeline_stage(size_t input_bank, size_t output_bank,
                                       const std::function<void()>& compute_func) {
    EDDOCommand compute_cmd{
        .phase = EDDOPhase::COMPUTE,
        .bank_id = input_bank,
        .source_addr = 0,
        .dest_addr = 0,
        .transfer_size = 0,
        .sequence_id = next_sequence_id++,
        .dependencies = {},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = [compute_func](const EDDOCommand&) { compute_func(); }
    };

    EDDOCommand writeback_cmd{
        .phase = EDDOPhase::WRITEBACK,
        .bank_id = output_bank,
        .source_addr = 0,
        .dest_addr = 0,
        .transfer_size = 0,
        .sequence_id = next_sequence_id++,
        .dependencies = {compute_cmd.sequence_id},
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX,
        .completion_callback = nullptr
    };

    enqueue_eddo_command(compute_cmd);
    enqueue_eddo_command(writeback_cmd);
}

size_t Buffet::get_pending_commands() const {
    std::lock_guard<std::mutex> lock(command_mutex);
    return command_queue.size();
}

bool Buffet::is_busy() const {
    std::lock_guard<std::mutex> lock(command_mutex);
    return !command_queue.empty() || !active_commands.empty();
}

bool Buffet::is_bank_busy(size_t bank_id) const {
    if (bank_id >= num_banks) return false;

    std::lock_guard<std::mutex> lock(bank_mutex);
    const auto& bank = *bank_states[bank_id];
    return bank.is_reading || bank.is_writing || bank.current_phase != EDDOPhase::SYNC;
}

Buffet::EDDOPhase Buffet::get_bank_phase(size_t bank_id) const {
    if (bank_id >= num_banks) return EDDOPhase::SYNC;

    std::lock_guard<std::mutex> lock(bank_mutex);
    return bank_states[bank_id]->current_phase;
}

Buffet::PerformanceMetrics Buffet::get_performance_metrics() const {
    std::lock_guard<std::mutex> lock(bank_mutex);

    PerformanceMetrics metrics{};
    size_t busy_banks = 0;

    for (const auto& bank : bank_states) {
        metrics.total_read_accesses += bank->read_accesses;
        metrics.total_write_accesses += bank->write_accesses;
        metrics.total_cache_hits += bank->cache_hits;
        metrics.total_cache_misses += bank->cache_misses;

        if (bank->current_phase != EDDOPhase::SYNC) {
            busy_banks++;
        }
    }

    metrics.average_bank_utilization = static_cast<double>(busy_banks) / num_banks;
    metrics.completed_eddo_commands = 0; // Would track this in real implementation

    return metrics;
}

Size Buffet::get_bank_capacity(size_t bank_id) const {
    if (bank_id >= num_banks) return 0;
    return bank_states[bank_id]->capacity;
}

Size Buffet::get_bank_occupancy(size_t bank_id) const {
    if (bank_id >= num_banks) return 0;
    return bank_states[bank_id]->current_occupancy;
}

void Buffet::reset() {
    std::lock_guard<std::mutex> cmd_lock(command_mutex);
    std::lock_guard<std::mutex> bank_lock(bank_mutex);

    // Clear command queues
    command_queue = std::queue<EDDOCommand>();
    active_commands.clear();
    dependency_graph.clear();

    // Reset bank states
    for (auto& bank : bank_states) {
        bank->current_occupancy = 0;
        bank->is_reading = false;
        bank->is_writing = false;
        bank->current_phase = EDDOPhase::SYNC;
        bank->active_sequence_id = 0;
        bank->read_accesses = 0;
        bank->write_accesses = 0;
        bank->cache_hits = 0;
        bank->cache_misses = 0;
        std::fill(bank->data.begin(), bank->data.end(), 0);
    }
}

void Buffet::flush_all_banks() {
    std::lock_guard<std::mutex> lock(bank_mutex);
    for (auto& bank : bank_states) {
        bank->current_occupancy = 0;
        std::fill(bank->data.begin(), bank->data.end(), 0);
    }
}

void Buffet::abort_pending_commands() {
    std::lock_guard<std::mutex> lock(command_mutex);
    command_queue = std::queue<EDDOCommand>();
    active_commands.clear();
    dependency_graph.clear();
}

// EDDOWorkflowBuilder Implementation
EDDOWorkflowBuilder& EDDOWorkflowBuilder::prefetch(size_t bank_id, Address src_addr,
                                                 Address dest_addr, Size size) {
    Buffet::EDDOCommand cmd{
        .phase = Buffet::EDDOPhase::PREFETCH,
        .bank_id = bank_id,
        .source_addr = src_addr,
        .dest_addr = dest_addr,
        .transfer_size = size,
        .sequence_id = next_sequence_id++,
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX
    };
    commands.push_back(cmd);
    return *this;
}

EDDOWorkflowBuilder& EDDOWorkflowBuilder::compute(size_t bank_id,
                                                const std::function<void()>& compute_func) {
    Buffet::EDDOCommand cmd{
        .phase = Buffet::EDDOPhase::COMPUTE,
        .bank_id = bank_id,
        .sequence_id = next_sequence_id++,
        .completion_callback = [compute_func](const Buffet::EDDOCommand&) { compute_func(); }
    };
    commands.push_back(cmd);
    return *this;
}

EDDOWorkflowBuilder& EDDOWorkflowBuilder::writeback(size_t bank_id, Address src_addr,
                                                  Address dest_addr, Size size) {
    Buffet::EDDOCommand cmd{
        .phase = Buffet::EDDOPhase::WRITEBACK,
        .bank_id = bank_id,
        .source_addr = src_addr,
        .dest_addr = dest_addr,
        .transfer_size = size,
        .sequence_id = next_sequence_id++,
        .block_mover_id = SIZE_MAX,
        .streamer_id = SIZE_MAX
    };
    commands.push_back(cmd);
    return *this;
}

EDDOWorkflowBuilder& EDDOWorkflowBuilder::sync() {
    Buffet::EDDOCommand cmd{
        .phase = Buffet::EDDOPhase::SYNC,
        .bank_id = 0, // Sync applies to all banks
        .sequence_id = next_sequence_id++
    };
    commands.push_back(cmd);
    return *this;
}

EDDOWorkflowBuilder& EDDOWorkflowBuilder::depend_on(size_t dependency_sequence_id) {
    if (!commands.empty()) {
        commands.back().dependencies.push_back(dependency_sequence_id);
    }
    return *this;
}

std::vector<Buffet::EDDOCommand> EDDOWorkflowBuilder::build() {
    return commands;
}

void EDDOWorkflowBuilder::execute_on(Buffet& buffet) {
    for (const auto& cmd : commands) {
        buffet.enqueue_eddo_command(cmd);
    }
}

} // namespace sw::kpu