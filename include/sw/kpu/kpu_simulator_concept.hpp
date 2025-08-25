#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <array>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <functional>

namespace sw::kpu {

// Forward declarations
class MainMemory;
class Scratchpad;
class DMAEngine;
class ComputeEngine;

// Memory address type
using Address = uint64_t;
using Size = uint64_t;

// Configuration constants
constexpr Size DEFAULT_MAIN_MEMORY_SIZE = 1ULL << 30; // 1GB
constexpr Size DEFAULT_SCRATCHPAD_SIZE = 1ULL << 20;  // 1MB
constexpr Size CACHE_LINE_SIZE = 64;

// Memory interface
class MemoryInterface {
public:
    virtual ~MemoryInterface() = default;
    virtual void read(Address addr, void* data, Size size) = 0;
    virtual void write(Address addr, const void* data, Size size) = 0;
    virtual Size size() const = 0;
};

// Main Memory implementation
class MainMemory : public MemoryInterface {
private:
    std::vector<uint8_t> memory_;
    mutable std::shared_mutex mutex_;

public:
    explicit MainMemory(Size size = DEFAULT_MAIN_MEMORY_SIZE) 
        : memory_(size, 0) {}

    void read(Address addr, void* data, Size size) override {
        std::shared_lock lock(mutex_);
        if (addr + size > memory_.size()) {
            throw std::out_of_range("Memory read out of bounds");
        }
        std::memcpy(data, memory_.data() + addr, size);
    }

    void write(Address addr, const void* data, Size size) override {
        std::unique_lock lock(mutex_);
        if (addr + size > memory_.size()) {
            throw std::out_of_range("Memory write out of bounds");
        }
        std::memcpy(memory_.data() + addr, data, size);
    }

    Size size() const override { 
        return memory_.size(); 
    }

    // Zero-copy access for performance-critical operations
    const uint8_t* data() const { return memory_.data(); }
    uint8_t* data() { return memory_.data(); }
};

// Software-managed scratchpad
class Scratchpad : public MemoryInterface {
private:
    std::vector<uint8_t> memory_;
    mutable std::mutex mutex_;

public:
    explicit Scratchpad(Size size = DEFAULT_SCRATCHPAD_SIZE) 
        : memory_(size, 0) {}

    void read(Address addr, void* data, Size size) override {
        std::lock_guard lock(mutex_);
        if (addr + size > memory_.size()) {
            throw std::out_of_range("Scratchpad read out of bounds");
        }
        std::memcpy(data, memory_.data() + addr, size);
    }

    void write(Address addr, const void* data, Size size) override {
        std::lock_guard lock(mutex_);
        if (addr + size > memory_.size()) {
            throw std::out_of_range("Scratchpad write out of bounds");
        }
        std::memcpy(memory_.data() + addr, data, size);
    }

    Size size() const override { 
        return memory_.size(); 
    }

    // High-performance direct access
    template<typename T>
    T* as() { return reinterpret_cast<T*>(memory_.data()); }
    
    template<typename T>
    const T* as() const { return reinterpret_cast<const T*>(memory_.data()); }
};

// DMA Transfer descriptor
struct DMATransfer {
    Address src_addr;
    Address dst_addr;
    Size size;
    bool src_is_main_memory;
    bool dst_is_main_memory;
    std::function<void()> completion_callback;
};

// DMA Engine
class DMAEngine {
private:
    MainMemory& main_memory_;
    Scratchpad& scratchpad_;
    std::queue<DMATransfer> transfer_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> running_;
    std::thread worker_thread_;

    void worker() {
        while (running_) {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !transfer_queue_.empty() || !running_; });
            
            if (!running_) break;
            
            auto transfer = std::move(transfer_queue_.front());
            transfer_queue_.pop();
            lock.unlock();
            
            execute_transfer(transfer);
        }
    }

    void execute_transfer(const DMATransfer& transfer) {
        std::vector<uint8_t> buffer(transfer.size);
        
        // Read from source
        if (transfer.src_is_main_memory) {
            main_memory_.read(transfer.src_addr, buffer.data(), transfer.size);
        } else {
            scratchpad_.read(transfer.src_addr, buffer.data(), transfer.size);
        }
        
        // Write to destination
        if (transfer.dst_is_main_memory) {
            main_memory_.write(transfer.dst_addr, buffer.data(), transfer.size);
        } else {
            scratchpad_.write(transfer.dst_addr, buffer.data(), transfer.size);
        }
        
        // Execute completion callback if provided
        if (transfer.completion_callback) {
            transfer.completion_callback();
        }
    }

public:
    DMAEngine(MainMemory& main_memory, Scratchpad& scratchpad)
        : main_memory_(main_memory), scratchpad_(scratchpad), running_(true) {
        worker_thread_ = std::thread(&DMAEngine::worker, this);
    }

    ~DMAEngine() {
        running_ = false;
        queue_cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }

    // Asynchronous DMA transfer
    void transfer_async(Address src_addr, Address dst_addr, Size size,
                       bool src_is_main_memory, bool dst_is_main_memory,
                       std::function<void()> callback = nullptr) {
        DMATransfer transfer{src_addr, dst_addr, size, 
                           src_is_main_memory, dst_is_main_memory, 
                           std::move(callback)};
        
        std::lock_guard lock(queue_mutex_);
        transfer_queue_.push(std::move(transfer));
        queue_cv_.notify_one();
    }

    // Synchronous DMA transfer
    void transfer_sync(Address src_addr, Address dst_addr, Size size,
                      bool src_is_main_memory, bool dst_is_main_memory) {
        std::mutex completion_mutex;
        std::condition_variable completion_cv;
        bool completed = false;
        
        auto callback = [&] {
            std::lock_guard lock(completion_mutex);
            completed = true;
            completion_cv.notify_one();
        };
        
        transfer_async(src_addr, dst_addr, size, 
                      src_is_main_memory, dst_is_main_memory, callback);
        
        std::unique_lock lock(completion_mutex);
        completion_cv.wait(lock, [&] { return completed; });
    }
};

// Matrix dimensions
struct MatrixDim {
    uint32_t rows;
    uint32_t cols;
};

// Compute Engine with Matrix Multiplication
class ComputeEngine {
private:
    Scratchpad& scratchpad_;
    
    // Simple GEMM implementation for simulation
    template<typename T>
    void gemm_impl(const T* A, const T* B, T* C, 
                   uint32_t M, uint32_t N, uint32_t K,
                   T alpha = T(1), T beta = T(0)) {
        // C = alpha * A * B + beta * C
        // A: M x K, B: K x N, C: M x N
        
        #pragma omp parallel for collapse(2) if(M * N > 1024)
        for (uint32_t i = 0; i < M; ++i) {
            for (uint32_t j = 0; j < N; ++j) {
                T sum = T(0);
                for (uint32_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = alpha * sum + beta * C[i * N + j];
            }
        }
    }

public:
    explicit ComputeEngine(Scratchpad& scratchpad) : scratchpad_(scratchpad) {}

    // Matrix multiplication: C = A * B
    void matmul_f32(Address addr_A, Address addr_B, Address addr_C,
                    MatrixDim dim_A, MatrixDim dim_B) {
        if (dim_A.cols != dim_B.rows) {
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
        }

        const auto* A = reinterpret_cast<const float*>(
            scratchpad_.as<uint8_t>() + addr_A);
        const auto* B = reinterpret_cast<const float*>(
            scratchpad_.as<uint8_t>() + addr_B);
        auto* C = reinterpret_cast<float*>(
            scratchpad_.as<uint8_t>() + addr_C);

        gemm_impl(A, B, C, dim_A.rows, dim_B.cols, dim_A.cols);
    }

    // Matrix multiplication with accumulation: C = A * B + C
    void matmul_accumulate_f32(Address addr_A, Address addr_B, Address addr_C,
                              MatrixDim dim_A, MatrixDim dim_B) {
        if (dim_A.cols != dim_B.rows) {
            throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
        }

        const auto* A = reinterpret_cast<const float*>(
            scratchpad_.as<uint8_t>() + addr_A);
        const auto* B = reinterpret_cast<const float*>(
            scratchpad_.as<uint8_t>() + addr_B);
        auto* C = reinterpret_cast<float*>(
            scratchpad_.as<uint8_t>() + addr_C);

        gemm_impl(A, B, C, dim_A.rows, dim_B.cols, dim_A.cols, 1.0f, 1.0f);
    }
};

// Main KPU Simulator class
class KPUSimulator {
private:
    std::unique_ptr<MainMemory> main_memory_;
    std::unique_ptr<Scratchpad> scratchpad_;
    std::unique_ptr<DMAEngine> dma_engine_;
    std::unique_ptr<ComputeEngine> compute_engine_;

public:
    explicit KPUSimulator(Size main_memory_size = DEFAULT_MAIN_MEMORY_SIZE,
                         Size scratchpad_size = DEFAULT_SCRATCHPAD_SIZE) {
        main_memory_ = std::make_unique<MainMemory>(main_memory_size);
        scratchpad_ = std::make_unique<Scratchpad>(scratchpad_size);
        dma_engine_ = std::make_unique<DMAEngine>(*main_memory_, *scratchpad_);
        compute_engine_ = std::make_unique<ComputeEngine>(*scratchpad_);
    }

    // Accessors
    MainMemory& main_memory() { return *main_memory_; }
    Scratchpad& scratchpad() { return *scratchpad_; }
    DMAEngine& dma_engine() { return *dma_engine_; }
    ComputeEngine& compute_engine() { return *compute_engine_; }

    const MainMemory& main_memory() const { return *main_memory_; }
    const Scratchpad& scratchpad() const { return *scratchpad_; }
    const DMAEngine& dma_engine() const { return *dma_engine_; }
    const ComputeEngine& compute_engine() const { return *compute_engine_; }
};

} // namespace sw::kpu