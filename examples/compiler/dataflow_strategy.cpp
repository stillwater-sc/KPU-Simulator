#include <sw/compiler/l2_tile_scheduler.hpp>
#include <sw/compiler/tile_optimizer.hpp>
#include <iostream>

using namespace sw::kpu::compiler;
using sw::kpu::Size;

int main() {
    TileOptimizer::MemoryHierarchy mem;
    TileOptimizer optimizer(mem);
    L2TileScheduler l2_scheduler;
    
    Size M = 512, N = 512, K = 512;
    auto tile_config = optimizer.optimize(M, N, K);
    
    std::cout << "Testing strategy-aware tile ordering:\n\n";
    
    // Test WS
    auto schedule_ws = l2_scheduler.generate_schedule(M, N, K, tile_config,
                                                      L2TileScheduler::ReplacementPolicy::LRU,
                                                      L2TileScheduler::SchedulingStrategy::WEIGHT_STATIONARY);
    std::cout << "WS: First 5 tile loads:\n";
    for (size_t i = 0; i < std::min(size_t(5), schedule_ws.load_sequence.size()); ++i) {
        auto& load = schedule_ws.load_sequence[i];
        std::cout << "  " << i << ": (ti=" << load.compute_ti 
                  << ", tj=" << load.compute_tj 
                  << ", tk=" << load.compute_tk << ")\n";
    }
    
    // Test IS
    auto schedule_is = l2_scheduler.generate_schedule(M, N, K, tile_config,
                                                      L2TileScheduler::ReplacementPolicy::LRU,
                                                      L2TileScheduler::SchedulingStrategy::INPUT_STATIONARY);
    std::cout << "IS: First 5 tile loads:\n";
    for (size_t i = 0; i < std::min(size_t(5), schedule_is.load_sequence.size()); ++i) {
        auto& load = schedule_is.load_sequence[i];
        std::cout << "  " << i << ": (ti=" << load.compute_ti 
                  << ", tj=" << load.compute_tj 
                  << ", tk=" << load.compute_tk << ")\n";
    }
    
    // Test OS
    auto schedule_os = l2_scheduler.generate_schedule(M, N, K, tile_config,
                                                      L2TileScheduler::ReplacementPolicy::LRU,
                                                      L2TileScheduler::SchedulingStrategy::OUTPUT_STATIONARY);
    std::cout << "\nOS: First 5 tile loads:\n";
    for (size_t i = 0; i < std::min(size_t(5), schedule_os.load_sequence.size()); ++i) {
        auto& load = schedule_os.load_sequence[i];
        std::cout << "  " << i << ": (ti=" << load.compute_ti 
                  << ", tj=" << load.compute_tj 
                  << ", tk=" << load.compute_tk << ")\n";
    }
    
    return 0;
}
