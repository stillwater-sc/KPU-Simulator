/**
 * @file autonomous_orchestrator.hpp
 * @brief Autonomous orchestration system for coordinating concurrent KPU components
 *
 * This orchestrator implements a signal-based synchronization system that allows
 * autonomous hardware components (DMA, BlockMover, Streamer, SystolicArray) to
 * collaborate without centralized control. Components signal completion of work,
 * and dependent operations automatically start when their dependencies are satisfied.
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace sw::sim {

/**
 * @brief Manages signal-based synchronization between autonomous hardware components
 *
 * The AutonomousOrchestrator allows programming a complete data flow pipeline upfront,
 * with explicit dependencies between stages. During execution, it automatically
 * launches operations when their dependencies are satisfied, modeling how real
 * hardware operates with hardware semaphores and synchronization flags.
 */
class AutonomousOrchestrator {
public:
    /**
     * @brief Represents an operation waiting for signal dependencies
     */
    struct PendingOperation {
        std::vector<std::string> required_signals;  // Signals that must be set
        std::function<void()> operation;             // Operation to execute
        std::string operation_name;                  // For debugging/logging
        bool executed;                               // Whether operation has run

        PendingOperation(const std::vector<std::string>& sigs,
                        std::function<void()> op,
                        const std::string& name = "")
            : required_signals(sigs), operation(std::move(op)),
              operation_name(name), executed(false) {}
    };

private:
    std::unordered_map<std::string, bool> signals_;
    std::vector<PendingOperation> pending_operations_;
    bool verbose_;
    size_t cycle_count_;

public:
    explicit AutonomousOrchestrator(bool verbose = false)
        : verbose_(verbose), cycle_count_(0) {}

    /**
     * @brief Signal that a component has completed its work
     *
     * @param name Signal name (e.g., "dma_input_done")
     */
    void signal(const std::string& name) {
        if (signals_[name]) {
            // Signal already set - this is fine, might happen in loops
            return;
        }

        signals_[name] = true;

        if (verbose_) {
            std::cout << "[Cycle " << std::setw(5) << cycle_count_
                      << "] Signal: " << name << "\n";
        }
    }

    /**
     * @brief Check if a signal has been set
     *
     * @param name Signal name
     * @return true if signal is set, false otherwise
     */
    bool is_signaled(const std::string& name) const {
        auto it = signals_.find(name);
        return it != signals_.end() && it->second;
    }

    /**
     * @brief Register an operation that waits for signals before executing
     *
     * @param required_signals Signals that must be set before operation can execute
     * @param operation Function to call when dependencies are satisfied
     * @param operation_name Optional name for debugging
     */
    void await(const std::vector<std::string>& required_signals,
               std::function<void()> operation,
               const std::string& operation_name = "") {
        pending_operations_.emplace_back(required_signals, std::move(operation), operation_name);
    }

    /**
     * @brief Convenience overload for single signal dependency
     *
     * @param required_signal Single signal to wait for
     * @param operation Function to call when dependency is satisfied
     * @param operation_name Optional name for debugging
     */
    void await(const std::string& required_signal,
               std::function<void()> operation,
               const std::string& operation_name = "") {
        await(std::vector<std::string>{required_signal}, std::move(operation), operation_name);
    }

    /**
     * @brief Check all pending operations and execute those whose dependencies are satisfied
     *
     * This should be called each simulation cycle to check if any new operations
     * can be launched based on newly signaled completions.
     *
     * @return Number of operations executed this step
     */
    size_t step() {
        cycle_count_++;
        size_t executed_count = 0;

        for (auto& op : pending_operations_) {
            if (op.executed) continue;

            // Check if all required signals are set
            bool all_satisfied = true;
            for (const auto& sig : op.required_signals) {
                if (!is_signaled(sig)) {
                    all_satisfied = false;
                    break;
                }
            }

            if (all_satisfied) {
                if (verbose_) {
                    std::cout << "[Cycle " << std::setw(5) << cycle_count_
                              << "] Launching: " << (op.operation_name.empty() ? "unnamed" : op.operation_name);
                    if (!op.required_signals.empty()) {
                        std::cout << " (waited for:";
                        for (const auto& sig : op.required_signals) {
                            std::cout << " " << sig;
                        }
                        std::cout << ")";
                    }
                    std::cout << "\n";
                }

                op.operation();
                op.executed = true;
                executed_count++;
            }
        }

        return executed_count;
    }

    /**
     * @brief Check if all pending operations have executed
     *
     * @return true if all operations complete, false otherwise
     */
    bool is_complete() const {
        return std::all_of(pending_operations_.begin(), pending_operations_.end(),
                          [](const auto& op) { return op.executed; });
    }

    /**
     * @brief Get the number of pending (not yet executed) operations
     *
     * @return Count of operations waiting or ready to execute
     */
    size_t get_pending_count() const {
        return std::count_if(pending_operations_.begin(), pending_operations_.end(),
                            [](const auto& op) { return !op.executed; });
    }

    /**
     * @brief Get the total number of operations (pending + completed)
     *
     * @return Total operation count
     */
    size_t get_total_operations() const {
        return pending_operations_.size();
    }

    /**
     * @brief Get the number of completed operations
     *
     * @return Count of executed operations
     */
    size_t get_completed_count() const {
        return std::count_if(pending_operations_.begin(), pending_operations_.end(),
                            [](const auto& op) { return op.executed; });
    }

    /**
     * @brief Get current cycle count
     */
    size_t get_cycle_count() const {
        return cycle_count_;
    }

    /**
     * @brief Reset the orchestrator state
     *
     * Clears all signals and pending operations, useful for running multiple tests
     */
    void reset() {
        signals_.clear();
        pending_operations_.clear();
        cycle_count_ = 0;
    }

    /**
     * @brief Print current status for debugging
     */
    void print_status() const {
        std::cout << "\n=== Orchestrator Status (Cycle " << cycle_count_ << ") ===\n";
        std::cout << "Total operations: " << get_total_operations() << "\n";
        std::cout << "Completed: " << get_completed_count() << "\n";
        std::cout << "Pending: " << get_pending_count() << "\n";

        std::cout << "\nActive signals:\n";
        for (const auto& [sig_name, is_set] : signals_) {
            if (is_set) {
                std::cout << "  ✓ " << sig_name << "\n";
            }
        }

        std::cout << "\nPending operations:\n";
        for (const auto& op : pending_operations_) {
            if (!op.executed) {
                std::cout << "  - " << (op.operation_name.empty() ? "unnamed" : op.operation_name);
                std::cout << " (waiting for:";
                bool first = true;
                for (const auto& sig : op.required_signals) {
                    if (!is_signaled(sig)) {
                        std::cout << (first ? " " : ", ") << sig;
                        first = false;
                    }
                }
                if (first) {
                    std::cout << " <ready to execute>";
                }
                std::cout << ")\n";
            }
        }
        std::cout << std::endl;
    }

    /**
     * @brief Enable or disable verbose logging
     */
    void set_verbose(bool verbose) {
        verbose_ = verbose;
    }
};

} // namespace sw::sim
