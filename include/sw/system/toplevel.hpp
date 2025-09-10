#pragma once

namespace sw::kpu {

/**
 * @brief Main System Simulator class
 * 
 * This is the primary interface for the functional simulator,
 * creating and coordinating all constituent components of a computing system.
 */
class TopLevelSimulator {
public:
    TopLevelSimulator();
    ~TopLevelSimulator() = default;
    
    // Non-copyable, movable
    TopLevelSimulator(const TopLevelSimulator&) = delete;
    TopLevelSimulator& operator=(const TopLevelSimulator&) = delete;
    TopLevelSimulator(TopLevelSimulator&&) = default;
    TopLevelSimulator& operator=(TopLevelSimulator&&) = default;
    
    /**
     * @brief Initialize the simulator
     * @return True if initialization successful
     */
    bool initialize();
    
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

private:
    bool initialized_{false};
    
    // TODO: Add other subsystems
    // DmaController dma_controller_;
    // BlockMover block_mover_;  
    // DataStreamer data_streamer_;
};

} // namespace sw::kpu