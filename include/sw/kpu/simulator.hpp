#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <stdexcept>
#include <iomanip>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251) // DLL interface warnings
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

namespace sw::kpu {

// Base address types
using Address = std::uint64_t;
using Size = std::size_t;
using Cycle = std::uint64_t;

// External memory component - manages its own memory model
class KPU_API ExternalMemory {
private:
    std::vector<std::uint8_t> memory_model;  // Dynamically managed resource
    Size capacity;
    Size bandwidth_bytes_per_cycle;
    mutable Cycle last_access_cycle;
    
public:
    explicit ExternalMemory(Size capacity_mb = 1024, Size bandwidth_gbps = 100);
    ~ExternalMemory() = default;
    
    // Core memory operations
    void read(Address addr, void* data, Size size);
    void write(Address addr, const void* data, Size size);
    bool is_ready() const;
    
    // Configuration and status
    Size get_capacity() const { return capacity; }
    Size get_bandwidth() const { return bandwidth_bytes_per_cycle; }
    void reset();
    
    // Statistics
    Cycle get_last_access_cycle() const { return last_access_cycle; }
};

// Software-managed scratchpad memory
class KPU_API Scratchpad {
private:
    std::vector<std::uint8_t> memory_model;  // Dynamically managed resource
    Size capacity;
    
public:
    explicit Scratchpad(Size capacity_kb = 512);
    ~Scratchpad() = default;
    
    // Core memory operations
    void read(Address addr, void* data, Size size);
    void write(Address addr, const void* data, Size size);
    bool is_ready() const { return true; } // Always ready - on-chip
    
    // Configuration and status
    Size get_capacity() const { return capacity; }
    void reset();
};

// DMA Engine for data movement between memory hierarchies
class KPU_API DMAEngine {
public:
    struct Transfer {
        Address src_addr;
        Address dst_addr;
        Size size;
        std::function<void()> completion_callback;
    };
    
    enum class MemoryType {
        EXTERNAL,
        SCRATCHPAD
    };
    
private:
    std::vector<Transfer> transfer_queue;  // Dynamically managed resource
    bool is_active;
    MemoryType src_type;
    MemoryType dst_type;
    size_t src_id;  // Index into memory bank vector
    size_t dst_id;  // Index into memory bank vector
    
public:
    DMAEngine(MemoryType src_type, size_t src_id, MemoryType dst_type, size_t dst_id);
    ~DMAEngine() = default;
    
    // Transfer operations
    void enqueue_transfer(Address src_addr, Address dst_addr, Size size, 
                         std::function<void()> callback = nullptr);
    bool process_transfers(std::vector<ExternalMemory>& memory_banks, 
                          std::vector<Scratchpad>& scratchpads);
    bool is_busy() const { return is_active || !transfer_queue.empty(); }
    void reset();
    
    // Configuration
    MemoryType get_src_type() const { return src_type; }
    MemoryType get_dst_type() const { return dst_type; }
    size_t get_src_id() const { return src_id; }
    size_t get_dst_id() const { return dst_id; }
};

// Compute fabric for matrix operations
class KPU_API ComputeFabric {
public:
    struct MatMulConfig {
        Size m, n, k; // Matrix dimensions: C[m,n] = A[m,k] * B[k,n]
        Address a_addr, b_addr, c_addr; // Addresses in scratchpad
        size_t scratchpad_id; // Which scratchpad to use
        std::function<void()> completion_callback;
    };
    
private:
    bool is_computing;
    Cycle compute_start_cycle;
    MatMulConfig current_op;
    size_t tile_id;  // Which compute tile this fabric represents
    
public:
    explicit ComputeFabric(size_t tile_id);
    ~ComputeFabric() = default;
    
    // Compute operations
    void start_matmul(const MatMulConfig& config);
    bool update(Cycle current_cycle, std::vector<Scratchpad>& scratchpads);
    bool is_busy() const { return is_computing; }
    void reset();
    
    // Configuration
    size_t get_tile_id() const { return tile_id; }
    
private:
    void execute_matmul(std::vector<Scratchpad>& scratchpads);
    Cycle estimate_cycles(Size m, Size n, Size k) const;
};

// Main KPU Simulator class - clean delegation-based API
class KPU_API KPUSimulator {
public:
    struct Config {
        Size memory_bank_count;
        Size memory_bank_capacity_mb;
        Size memory_bandwidth_gbps;
        Size scratchpad_count;
        Size scratchpad_capacity_kb;
        Size compute_tile_count;
        Size dma_engine_count;

		Config() = default;
		Config(const Config&) = default;
		Config& operator=(const Config&) = default;
		Config(Config&&) = default;
		Config& operator=(Config&&) = default;
		~Config() = default;

        Config (Size mem_banks, Size mem_cap, Size mem_bw,
                Size pads, Size pad_cap,
                Size tiles, Size dmas)
            : memory_bank_count(mem_banks), memory_bank_capacity_mb(mem_cap),
              memory_bandwidth_gbps(mem_bw), scratchpad_count(pads),
              scratchpad_capacity_kb(pad_cap), compute_tile_count(tiles),
			dma_engine_count(dmas) {
		}
    };
    
    struct MatMulTest {
        Size m, n, k;
        std::vector<float> matrix_a;
        std::vector<float> matrix_b;
        std::vector<float> expected_c;
    };
    
private:
    // Component vectors - value semantics, addressable
    std::vector<ExternalMemory> memory_banks;
    std::vector<Scratchpad> scratchpads;
    std::vector<DMAEngine> dma_engines;
    std::vector<ComputeFabric> compute_tiles;
    
    // Simulation state
    Cycle current_cycle;
    std::chrono::high_resolution_clock::time_point sim_start_time;
    
public:
    explicit KPUSimulator(const Config& config = {});  // Config{ 2,1024,100,2,64,2,2 }: 2 banks, 1GB each, 100GBps, 2 pads 64KB each, 2 tiles, 2 DMAs
    ~KPUSimulator() = default;
    
    // Memory operations - clean delegation API
    void read_memory_bank(size_t bank_id, Address addr, void* data, Size size);
    void write_memory_bank(size_t bank_id, Address addr, const void* data, Size size);
    void read_scratchpad(size_t pad_id, Address addr, void* data, Size size);
    void write_scratchpad(size_t pad_id, Address addr, const void* data, Size size);
    
    // DMA operations
    void start_dma_transfer(size_t dma_id, Address src_addr, Address dst_addr, 
                           Size size, std::function<void()> callback = nullptr);
    bool is_dma_busy(size_t dma_id);
    
    // Compute operations
    void start_matmul(size_t tile_id, size_t scratchpad_id, Size m, Size n, Size k,
                     Address a_addr, Address b_addr, Address c_addr,
                     std::function<void()> callback = nullptr);
    bool is_compute_busy(size_t tile_id);
    
    // Simulation control
    void reset();
    void step(); // Single simulation step
    void run_until_idle(); // Run until all components are idle
    
    // Configuration queries
    size_t get_memory_bank_count() const { return memory_banks.size(); }
    size_t get_scratchpad_count() const { return scratchpads.size(); }
    size_t get_compute_tile_count() const { return compute_tiles.size(); }
    size_t get_dma_engine_count() const { return dma_engines.size(); }
    
    Size get_memory_bank_capacity(size_t bank_id) const;
    Size get_scratchpad_capacity(size_t pad_id) const;
    
    // High-level test operations
    bool run_matmul_test(const MatMulTest& test, size_t memory_bank_id = 0, 
                        size_t scratchpad_id = 0, size_t compute_tile_id = 0);
    
    // Statistics and monitoring
    Cycle get_current_cycle() const { return current_cycle; }
    double get_elapsed_time_ms() const;
    void print_stats() const;
    void print_component_status() const;
    
    // Component status queries
    bool is_memory_bank_ready(size_t bank_id) const;
    bool is_scratchpad_ready(size_t pad_id) const;
    
private:
    void validate_bank_id(size_t bank_id) const;
    void validate_scratchpad_id(size_t pad_id) const;
    void validate_dma_id(size_t dma_id) const;
    void validate_tile_id(size_t tile_id) const;
};

// Utility functions for test case generation
namespace test_utils {
    KPU_API KPUSimulator::MatMulTest generate_simple_matmul_test(Size m = 4, Size n = 4, Size k = 4);
    KPU_API std::vector<float> generate_random_matrix(Size rows, Size cols, float min_val = -1.0f, float max_val = 1.0f);
    KPU_API bool verify_matmul_result(const std::vector<float>& a, const std::vector<float>& b, 
                             const std::vector<float>& c, Size m, Size n, Size k, float tolerance = 1e-5f);
    
    // Multi-bank test utilities
    KPU_API KPUSimulator::Config generate_multi_bank_config(size_t num_banks = 4, size_t num_tiles = 2);
    KPU_API bool run_distributed_matmul_test(KPUSimulator& sim, Size matrix_size = 8);
}

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif