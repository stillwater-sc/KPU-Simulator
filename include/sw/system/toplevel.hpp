#pragma once

#include "system_config.hpp"
#include <memory>
#include <vector>
#include <filesystem>

// Forward declarations
namespace sw::kpu {
    class KPUSimulator;
}

namespace sw::sim {

/**
 * @brief Main System Simulator class
 *
 * This is the primary interface for the functional simulator,
 * creating and coordinating all constituent components of a computing system.
 * The system is configured via JSON files or programmatic SystemConfig objects.
 */
class SystemSimulator {
public:
    /**
     * @brief Construct with default configuration
     */
    SystemSimulator();

    /**
     * @brief Construct with specific configuration
     * @param config System configuration
     */
    explicit SystemSimulator(const SystemConfig& config);

    /**
     * @brief Construct and load configuration from JSON file
     * @param config_file Path to JSON configuration file
     */
    explicit SystemSimulator(const std::filesystem::path& config_file);

    ~SystemSimulator();

    // Non-copyable, movable
    SystemSimulator(const SystemSimulator&) = delete;
    SystemSimulator& operator=(const SystemSimulator&) = delete;
    SystemSimulator(SystemSimulator&&) noexcept;
    SystemSimulator& operator=(SystemSimulator&&) noexcept;

    /**
     * @brief Initialize the simulator with current configuration
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * @brief Initialize with new configuration
     * @param config System configuration
     * @return True if initialization successful
     */
    bool initialize(const SystemConfig& config);

    /**
     * @brief Load configuration from JSON file and initialize
     * @param config_file Path to JSON configuration file
     * @return True if initialization successful
     */
    bool load_config_and_initialize(const std::filesystem::path& config_file);

    /**
     * @brief Shutdown the simulator and clean up resources
     */
    void shutdown();

    /**
     * @brief Check if simulator is initialized
     * @return True if ready for operations
     */
    bool is_initialized() const noexcept { return initialized_; }

    /**
     * @brief Run a simple test operation
     * @return True if test passed
     */
    bool run_self_test();

    /**
     * @brief Get current configuration
     * @return Reference to system configuration
     */
    const SystemConfig& get_config() const { return config_; }

    /**
     * @brief Get number of KPU accelerators
     * @return KPU count
     */
    size_t get_kpu_count() const;

    /**
     * @brief Get KPU simulator by index
     * @param index KPU index
     * @return Pointer to KPU simulator (nullptr if invalid index)
     */
    sw::kpu::KPUSimulator* get_kpu(size_t index);

    /**
     * @brief Get KPU simulator by ID
     * @param id Accelerator ID from configuration
     * @return Pointer to KPU simulator (nullptr if not found)
     */
    sw::kpu::KPUSimulator* get_kpu_by_id(const std::string& id);

    /**
     * @brief Print system configuration summary
     */
    void print_config() const;

    /**
     * @brief Print system status
     */
    void print_status() const;

private:
    bool initialized_{false};
    SystemConfig config_;

    // Component instances (created based on configuration)
    std::vector<std::unique_ptr<sw::kpu::KPUSimulator>> kpu_instances_;

    // Helper methods
    void create_components_from_config();
    void destroy_components();
    sw::kpu::KPUSimulator* create_kpu_from_config(const KPUConfig& kpu_config);
};

} // namespace sw::sim