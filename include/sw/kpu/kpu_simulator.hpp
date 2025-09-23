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

#include <sw/concepts.hpp>
#include <sw/memory/external_memory.hpp>
#include <sw/kpu/scratchpad.hpp>
#include <sw/kpu/dma_engine.hpp>
#include <sw/kpu/l3_tile.hpp>
#include <sw/kpu/l2_bank.hpp>
#include <sw/kpu/block_mover.hpp>
#include <sw/kpu/compute_fabric.hpp>
namespace sw::kpu {

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
        Size l3_tile_count;
        Size l3_tile_capacity_kb;
        Size l2_bank_count;
        Size l2_bank_capacity_kb;
        Size block_mover_count;

        Config() : memory_bank_count(2), memory_bank_capacity_mb(1024),
                   memory_bandwidth_gbps(100), scratchpad_count(2),
                   scratchpad_capacity_kb(64), compute_tile_count(2),
                   dma_engine_count(2), l3_tile_count(4), l3_tile_capacity_kb(128),
                   l2_bank_count(8), l2_bank_capacity_kb(64), block_mover_count(4) {}
		Config(const Config&) = default;
		Config& operator=(const Config&) = default;
		Config(Config&&) = default;
		Config& operator=(Config&&) = default;
		~Config() = default;

        Config (Size mem_banks, Size mem_cap, Size mem_bw,
                Size pads, Size pad_cap,
                Size tiles, Size dmas, Size l3_tiles = 4, Size l3_cap = 128,
                Size l2_banks = 8, Size l2_cap = 64, Size block_movers = 4)
            : memory_bank_count(mem_banks), memory_bank_capacity_mb(mem_cap),
              memory_bandwidth_gbps(mem_bw), scratchpad_count(pads),
              scratchpad_capacity_kb(pad_cap), compute_tile_count(tiles),
              dma_engine_count(dmas), l3_tile_count(l3_tiles), l3_tile_capacity_kb(l3_cap),
              l2_bank_count(l2_banks), l2_bank_capacity_kb(l2_cap), block_mover_count(block_movers) {
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
    std::vector<L3Tile> l3_tiles;
    std::vector<L2Bank> l2_banks;
    std::vector<BlockMover> block_movers;
    
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
    
    // DMA operations - now bidirectional with per-transfer configuration
    void start_dma_transfer(size_t dma_id,
                           DMAEngine::MemoryType src_type, size_t src_id, Address src_addr,
                           DMAEngine::MemoryType dst_type, size_t dst_id, Address dst_addr,
                           Size size, std::function<void()> callback = nullptr);

    // Convenience methods for common transfer patterns
    void start_dma_external_to_scratchpad(size_t dma_id, size_t bank_id, Address src_addr,
                                          size_t pad_id, Address dst_addr, Size size,
                                          std::function<void()> callback = nullptr);
    void start_dma_scratchpad_to_external(size_t dma_id, size_t pad_id, Address src_addr,
                                          size_t bank_id, Address dst_addr, Size size,
                                          std::function<void()> callback = nullptr);

    bool is_dma_busy(size_t dma_id);

    // BlockMover operations - L3 to L2 data movement with transformations
    void start_block_transfer(size_t block_mover_id, size_t src_l3_tile_id, Address src_offset,
                             size_t dst_l2_bank_id, Address dst_offset,
                             Size block_height, Size block_width, Size element_size,
                             BlockMover::TransformType transform = BlockMover::TransformType::IDENTITY,
                             std::function<void()> callback = nullptr);

    bool is_block_mover_busy(size_t block_mover_id);

    // L3 and L2 memory operations
    void read_l3_tile(size_t tile_id, Address addr, void* data, Size size);
    void write_l3_tile(size_t tile_id, Address addr, const void* data, Size size);
    void read_l2_bank(size_t bank_id, Address addr, void* data, Size size);
    void write_l2_bank(size_t bank_id, Address addr, const void* data, Size size);

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
    size_t get_l3_tile_count() const { return l3_tiles.size(); }
    size_t get_l2_bank_count() const { return l2_banks.size(); }
    size_t get_block_mover_count() const { return block_movers.size(); }
    
    Size get_memory_bank_capacity(size_t bank_id) const;
    Size get_scratchpad_capacity(size_t pad_id) const;
    Size get_l3_tile_capacity(size_t tile_id) const;
    Size get_l2_bank_capacity(size_t bank_id) const;
    
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
    bool is_l3_tile_ready(size_t tile_id) const;
    bool is_l2_bank_ready(size_t bank_id) const;
    
private:
    void validate_bank_id(size_t bank_id) const;
    void validate_scratchpad_id(size_t pad_id) const;
    void validate_dma_id(size_t dma_id) const;
    void validate_tile_id(size_t tile_id) const;
    void validate_l3_tile_id(size_t tile_id) const;
    void validate_l2_bank_id(size_t bank_id) const;
    void validate_block_mover_id(size_t mover_id) const;
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