#include "sw/kpu/simulator.hpp"
#include <algorithm>
#include <random>
#include <iostream>
#include <cassert>
#include <cstring>

namespace sw::kpu {

// ExternalMemory implementation
ExternalMemory::ExternalMemory(Size capacity_mb, Size bandwidth_gbps) 
    : capacity_(capacity_mb * 1024 * 1024), 
      bandwidth_bytes_per_cycle_(bandwidth_gbps * 1000000000 / 8 / 1000000000), // Assuming 1GHz clock
      last_access_cycle_(0) {
    memory_.resize(capacity_);
    std::fill(memory_.begin(), memory_.end(), 0);
}

void ExternalMemory::read(Address addr, void* data, Size size) {
    if (addr + size > capacity_) {
        throw std::out_of_range("Memory read out of bounds");
    }
    std::memcpy(data, memory_.data() + addr, size);
}

void ExternalMemory::write(Address addr, const void* data, Size size) {
    if (addr + size > capacity_) {
        throw std::out_of_range("Memory write out of bounds");
    }
    std::memcpy(memory_.data() + addr, data, size);
}

bool ExternalMemory::is_ready() const {
    // Simplified: assume always ready for now
    return true;
}

void ExternalMemory::reset() {
    std::fill(memory_.begin(), memory_.end(), 0);
    last_access_cycle_ = 0;
}

// DMAEngine implementation
DMAEngine::DMAEngine(MemoryInterface* src, MemoryInterface* dst)
    : src_memory_(src), dst_memory_(dst), is_active_(false) {}

void DMAEngine::enqueue_transfer(Address src_addr, Address dst_addr, Size size, 
                                std::function<void()> callback) {
    transfer_queue_.emplace_back(Transfer{src_addr, dst_addr, size, std::move(callback)});
}

bool DMAEngine::process_transfers() {
    if (transfer_queue_.empty()) {
        is_active_ = false;
        return false;
    }
    
    is_active_ = true;
    auto& transfer = transfer_queue_.front();
    
    // Allocate temporary buffer for the transfer
    std::vector<std::uint8_t> buffer(transfer.size);
    
    // Read from source
    src_memory_->read(transfer.src_addr, buffer.data(), transfer.size);
    
    // Write to destination
    dst_memory_->write(transfer.dst_addr, buffer.data(), transfer.size);
    
    // Call completion callback if provided
    if (transfer.completion_callback) {
        transfer.completion_callback();
    }
    
    transfer_queue_.erase(transfer_queue_.begin());
    
    bool completed = transfer_queue_.empty();
    if (completed) {
        is_active_ = false;
    }
    
    return completed;
}

void DMAEngine::reset() {
    transfer_queue_.clear();
    is_active_ = false;
}

// Scratchpad implementation
Scratchpad::Scratchpad(Size capacity_kb) 
    : capacity_(capacity_kb * 1024) {
    memory_.resize(capacity_);
    std::fill(memory_.begin(), memory_.end(), 0);
}

void Scratchpad::read(Address addr, void* data, Size size) {
    if (addr + size > capacity_) {
        throw std::out_of_range("Scratchpad read out of bounds");
    }
    std::memcpy(data, memory_.data() + addr, size);
}

void Scratchpad::write(Address addr, const void* data, Size size) {
    if (addr + size > capacity_) {
        throw std::out_of_range("Scratchpad write out of bounds");
    }
    std::memcpy(memory_.data() + addr, data, size);
}

void Scratchpad::reset() {
    std::fill(memory_.begin(), memory_.end(), 0);
}

// ComputeFabric implementation
ComputeFabric::ComputeFabric(Scratchpad* scratchpad)
    : scratchpad_(scratchpad), is_computing_(false), compute_start_cycle_(0) {}

void ComputeFabric::start_matmul(const MatMulConfig& config) {
    if (is_computing_) {
        throw std::runtime_error("ComputeFabric is already busy");
    }
    
    current_op_ = config;
    is_computing_ = true;
    compute_start_cycle_ = 0; // Will be set by the caller
}

bool ComputeFabric::update(Cycle current_cycle) {
    if (!is_computing_) {
        return false;
    }
    
    if (compute_start_cycle_ == 0) {
        compute_start_cycle_ = current_cycle;
    }
    
    Cycle required_cycles = estimate_cycles(current_op_.m, current_op_.n, current_op_.k);
    
    if (current_cycle - compute_start_cycle_ >= required_cycles) {
        // Operation completed
        execute_matmul();
        
        if (current_op_.completion_callback) {
            current_op_.completion_callback();
        }
        
        is_computing_ = false;
        return true;
    }
    
    return false;
}

void ComputeFabric::execute_matmul() {
    // Read matrices from scratchpad
    Size a_size = current_op_.m * current_op_.k * sizeof(float);
    Size b_size = current_op_.k * current_op_.n * sizeof(float);
    Size c_size = current_op_.m * current_op_.n * sizeof(float);
    
    std::vector<float> a(current_op_.m * current_op_.k);
    std::vector<float> b(current_op_.k * current_op_.n);
    std::vector<float> c(current_op_.m * current_op_.n, 0.0f);
    
    scratchpad_->read(current_op_.a_addr, a.data(), a_size);
    scratchpad_->read(current_op_.b_addr, b.data(), b_size);
    
    // Perform matrix multiplication: C = A * B
    for (Size i = 0; i < current_op_.m; ++i) {
        for (Size j = 0; j < current_op_.n; ++j) {
            float sum = 0.0f;
            for (Size k = 0; k < current_op_.k; ++k) {
                sum += a[i * current_op_.k + k] * b[k * current_op_.n + j];
            }
            c[i * current_op_.n + j] = sum;
        }
    }
    
    // Write result back to scratchpad
    scratchpad_->write(current_op_.c_addr, c.data(), c_size);
}

Cycle ComputeFabric::estimate_cycles(Size m, Size n, Size k) const {
    // Simplified model: assume 1 cycle per MAC operation
    return m * n * k;
}

void ComputeFabric::reset() {
    is_computing_ = false;
    compute_start_cycle_ = 0;
}

// KPUSimulator implementation
KPUSimulator::KPUSimulator(const Config& config) : current_cycle_(0) {
    // Initialize components
    external_memory_ = std::make_unique<ExternalMemory>(config.external_memory_mb, config.memory_bandwidth_gbps);
    scratchpad_ = std::make_unique<Scratchpad>(config.scratchpad_kb);
    
    dma_ext_to_scratch_ = std::make_unique<DMAEngine>(external_memory_.get(), scratchpad_.get());
    dma_scratch_to_ext_ = std::make_unique<DMAEngine>(scratchpad_.get(), external_memory_.get());
    
    compute_fabric_ = std::make_unique<ComputeFabric>(scratchpad_.get());
    
    sim_start_time_ = std::chrono::high_resolution_clock::now();
}

void KPUSimulator::reset() {
    external_memory_->reset();
    scratchpad_->reset();
    dma_ext_to_scratch_->reset();
    dma_scratch_to_ext_->reset();
    compute_fabric_->reset();
    current_cycle_ = 0;
    sim_start_time_ = std::chrono::high_resolution_clock::now();
}

void KPUSimulator::step() {
    ++current_cycle_;
    
    // Update all components
    dma_ext_to_scratch_->process_transfers();
    dma_scratch_to_ext_->process_transfers();
    compute_fabric_->update(current_cycle_);
}

void KPUSimulator::run_until_idle() {
    while (dma_ext_to_scratch_->is_busy() || 
           dma_scratch_to_ext_->is_busy() || 
           compute_fabric_->is_busy()) {
        step();
    }
}

bool KPUSimulator::run_matmul_test(const MatMulTest& test) {
    reset();
    
    Size a_size = test.m * test.k * sizeof(float);
    Size b_size = test.k * test.n * sizeof(float);
    Size c_size = test.m * test.n * sizeof(float);
    
    // Addresses in external memory
    Address ext_a_addr = 0;
    Address ext_b_addr = a_size;
    Address ext_c_addr = ext_b_addr + b_size;
    
    // Addresses in scratchpad
    Address scratch_a_addr = 0;
    Address scratch_b_addr = a_size;
    Address scratch_c_addr = scratch_b_addr + b_size;
    
    try {
        // Load test data into external memory
        external_memory_->write(ext_a_addr, test.matrix_a.data(), a_size);
        external_memory_->write(ext_b_addr, test.matrix_b.data(), b_size);
        
        // Set up computation pipeline
        bool dma_a_complete = false, dma_b_complete = false, compute_complete = false;
        
        // DMA A matrix to scratchpad
        dma_ext_to_scratch_->enqueue_transfer(ext_a_addr, scratch_a_addr, a_size, 
            [&dma_a_complete]() { dma_a_complete = true; });
        
        // DMA B matrix to scratchpad
        dma_ext_to_scratch_->enqueue_transfer(ext_b_addr, scratch_b_addr, b_size, 
            [&dma_b_complete]() { dma_b_complete = true; });
        
        // Wait for data to be loaded
        while (!dma_a_complete || !dma_b_complete) {
            step();
        }
        
        // Start matrix multiplication
        ComputeFabric::MatMulConfig matmul_config{
            .m = test.m, .n = test.n, .k = test.k,
            .a_addr = scratch_a_addr, .b_addr = scratch_b_addr, .c_addr = scratch_c_addr,
            .completion_callback = [&compute_complete]() { compute_complete = true; }
        };
        
        compute_fabric_->start_matmul(matmul_config);
        
        // Wait for computation to complete
        while (!compute_complete) {
            step();
        }
        
        // DMA result back to external memory
        bool dma_c_complete = false;
        dma_scratch_to_ext_->enqueue_transfer(scratch_c_addr, ext_c_addr, c_size,
            [&dma_c_complete]() { dma_c_complete = true; });
        
        // Wait for result transfer
        while (!dma_c_complete) {
            step();
        }
        
        // Verify result
        std::vector<float> result_c(test.m * test.n);
        external_memory_->read(ext_c_addr, result_c.data(), c_size);
        
        return test_utils::verify_matmul_result(test.matrix_a, test.matrix_b, result_c, 
                                               test.m, test.n, test.k);
    }
    catch (const std::exception& e) {
        std::cerr << "Error during matmul test: " << e.what() << std::endl;
        return false;
    }
}

double KPUSimulator::get_elapsed_time_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - sim_start_time_);
    return duration.count() / 1000.0;
}

void KPUSimulator::print_stats() const {
    std::cout << "=== KPU Simulator Statistics ===" << std::endl;
    std::cout << "Simulation cycles: " << current_cycle_ << std::endl;
    std::cout << "Wall-clock time: " << get_elapsed_time_ms() << " ms" << std::endl;
    std::cout << "External memory capacity: " << external_memory_->get_capacity() / (1024*1024) << " MB" << std::endl;
    std::cout << "Scratchpad capacity: " << scratchpad_->get_capacity() / 1024 << " KB" << std::endl;
}

// Test utilities implementation
namespace test_utils {

KPUSimulator::MatMulTest generate_simple_matmul_test(Size m, Size n, Size k) {
    KPUSimulator::MatMulTest test;
    test.m = m;
    test.n = n; 
    test.k = k;
    
    test.matrix_a = generate_random_matrix(m, k, -2.0f, 2.0f);
    test.matrix_b = generate_random_matrix(k, n, -2.0f, 2.0f);
    
    // Compute expected result
    test.expected_c.resize(m * n);
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (Size ki = 0; ki < k; ++ki) {
                sum += test.matrix_a[i * k + ki] * test.matrix_b[ki * n + j];
            }
            test.expected_c[i * n + j] = sum;
        }
    }
    
    return test;
}

std::vector<float> generate_random_matrix(Size rows, Size cols, float min_val, float max_val) {
    std::vector<float> matrix(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    std::generate(matrix.begin(), matrix.end(), [&]() { return dis(gen); });
    return matrix;
}

bool verify_matmul_result(const std::vector<float>& a, const std::vector<float>& b, 
                         const std::vector<float>& c, Size m, Size n, Size k, float tolerance) {
    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            float expected = 0.0f;
            for (Size ki = 0; ki < k; ++ki) {
                expected += a[i * k + ki] * b[ki * n + j];
            }
            
            float actual = c[i * n + j];
            if (std::abs(actual - expected) > tolerance) {
                std::cerr << "Mismatch at (" << i << "," << j << "): expected " 
                         << expected << ", got " << actual << std::endl;
                return false;
            }
        }
    }
    return true;
}

} // namespace test_utils

} // namespace sw::kpu