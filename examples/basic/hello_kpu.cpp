/**
 * @file hello_kpu.cpp
 * @brief Simple first KPU program - Hello World for KPU simulator
 */

#include <sw/kpu/kpu_simulator.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "===========================================\n";
    std::cout << " Hello KPU - First KPU Program\n";
    std::cout << "===========================================\n\n";

    // Create a simple KPU configuration
    sw::kpu::KPUSimulator::Config config(
        2,      // 2 memory banks
        1024,   // 1GB each
        100,    // 100 GB/s bandwidth
        2,      // 2 scratchpads
        64,     // 64KB each
        2,      // 2 compute tiles
        2       // 2 DMA engines
    );

    std::cout << "Creating KPU with configuration:\n";
    std::cout << "  Memory banks: " << config.memory_bank_count << "\n";
    std::cout << "  Scratchpads: " << config.scratchpad_count << "\n";
    std::cout << "  Compute tiles: " << config.compute_tile_count << "\n\n";

    // Create simulator
    sw::kpu::KPUSimulator kpu(config);

    std::cout << "KPU created successfully!\n";
    std::cout << "  Using systolic arrays: " << (kpu.is_using_systolic_arrays() ? "Yes" : "No") << "\n";

    if (kpu.is_using_systolic_arrays()) {
        std::cout << "  Systolic array size: "
                  << kpu.get_systolic_array_rows() << "x"
                  << kpu.get_systolic_array_cols() << "\n";
        std::cout << "  Total PEs: " << kpu.get_systolic_array_total_pes() << "\n";
    }

    std::cout << "\n===========================================\n";
    std::cout << " KPU is ready for computation!\n";
    std::cout << "===========================================\n";

    return 0;
}
