#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <chrono>

namespace sw::kpu {

// Forward declarations
class ExternalMemory;
class DMAEngine;
class Scratchpad;
class ComputeFabric;

// Base address types
using Address = std::uint64_t;
using Size = std::size_t;
using Cycle = std::uint64_t;

// Memory interface for all components
class MemoryInterface {
public:
    virtual ~MemoryInterface() = default;
    virtual void read(Address addr, void* data, Size size) = 0;
    virtual void write(Address addr, const void* data, Size size) = 0;
    virtual bool is_ready() const = 0;
};

// External memory component - DDR/HBM simulation
class ExternalMemory : public MemoryInterface {
private:
    std::vector<std::uint8_t> memory_;
    Size capacity_;
    Size bandwidth_bytes_per_cycle_;
    mutable Cycle last_access_cycle_;
    
public:
    explicit ExternalMemory(Size capacity_mb = 1024, Size bandwidth_gbps = 100);
    
    void read(Address addr, void* data, Size size) override;
    void write(Address addr, const void* data, Size size) override;
    bool is_ready() const override;
    
    Size get_capacity() const { return capacity_; }
    void reset();
};

// DMA Engine for data movement
class DMAEngine {
public:
    struct Transfer {
        Address src_addr;
        Address dst_addr;
        Size size;
        std::function<void()> completion_callback;
    };
    
private:
    MemoryInterface* src_memory_;
    MemoryInterface* dst_memory_;
    std::vector<Transfer> transfer_queue_;
    bool is_active_;
    
public:
    DMAEngine(MemoryInterface* src, MemoryInterface* dst);
    
    void enqueue_transfer(Address src_addr, Address dst_addr, Size size, 
                         std::function<void()> callback = nullptr);
    bool process_transfers(); // Returns true if transfers completed
    bool is_busy() const { return is_active_ || !transfer_queue_.empty(); }
    void reset();
};

// Software-managed scratchpad memory
class Scratchpad : public MemoryInterface {
private:
    std::vector<std::uint8_t> memory_;
    Size capacity_;
    
public:
    explicit Scratchpad(Size capacity_kb = 512);
    
    void read(Address addr, void* data, Size size) override;
    void write(Address addr, const void* data, Size size) override;
    bool is_ready() const override { return true; } // Always ready - on-chip
    
    Size get_capacity() const { return capacity_; }
    void reset();
};

// Compute fabric for matrix operations
class ComputeFabric {
public:
    struct MatMulConfig {
        Size m, n, k; // Matrix dimensions: C[m,n] = A[m,k] * B[k,n]
        Address a_addr, b_addr, c_addr; // Addresses in scratchpad
        std::function<void()> completion_callback;
    };
    
private:
    Scratchpad* scratchpad_;
    bool is_computing_;
    Cycle compute_start_cycle_;
    MatMulConfig current_op_;
    
public:
    explicit ComputeFabric(Scratchpad* scratchpad);
    
    void start_matmul(const MatMulConfig& config);
    bool is_busy() const { return is_computing_; }
    bool update(Cycle current_cycle); // Returns true if operation completed
    void reset();
    
private:
    void execute_matmul();
    Cycle estimate_cycles(Size m, Size n, Size k) const;
};

// Main KPU Simulator class
class KPUSimulator {
private:
    // Components
    std::unique_ptr<ExternalMemory> external_memory_;
    std::unique_ptr<Scratchpad> scratchpad_;
    std::unique_ptr<DMAEngine> dma_ext_to_scratch_;
    std::unique_ptr<DMAEngine> dma_scratch_to_ext_;
    std::unique_ptr<ComputeFabric> compute_fabric_;
    
    // Simulation state
    Cycle current_cycle_;
    std::chrono::high_resolution_clock::time_point sim_start_time_;
    
public:
    struct Config {
        Size external_memory_mb = 1024;
        Size scratchpad_kb = 512;
        Size memory_bandwidth_gbps = 100;
    };
    
    explicit KPUSimulator(const Config& config = {});
    ~KPUSimulator() = default;
    
    // Component access
    ExternalMemory* get_external_memory() { return external_memory_.get(); }
    Scratchpad* get_scratchpad() { return scratchpad_.get(); }
    DMAEngine* get_dma_ext_to_scratch() { return dma_ext_to_scratch_.get(); }
    DMAEngine* get_dma_scratch_to_ext() { return dma_scratch_to_ext_.get(); }
    ComputeFabric* get_compute_fabric() { return compute_fabric_.get(); }
    
    // Simulation control
    void reset();
    void step(); // Single simulation step
    void run_until_idle(); // Run until all components are idle
    
    // Matrix multiplication test case
    struct MatMulTest {
        Size m, n, k;
        std::vector<float> matrix_a;
        std::vector<float> matrix_b;
        std::vector<float> expected_c;
    };
    
    bool run_matmul_test(const MatMulTest& test);
    
    // Statistics
    Cycle get_current_cycle() const { return current_cycle_; }
    double get_elapsed_time_ms() const;
    void print_stats() const;
};

// Utility functions for test case generation
namespace test_utils {
    KPUSimulator::MatMulTest generate_simple_matmul_test(Size m = 4, Size n = 4, Size k = 4);
    std::vector<float> generate_random_matrix(Size rows, Size cols, float min_val = -1.0f, float max_val = 1.0f);
    bool verify_matmul_result(const std::vector<float>& a, const std::vector<float>& b, 
                             const std::vector<float>& c, Size m, Size n, Size k, float tolerance = 1e-5f);
}

} // namespace sw::kpu