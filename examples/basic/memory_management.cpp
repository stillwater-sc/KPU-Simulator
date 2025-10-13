/**
 * @file memory_management.cpp
 * @brief Demonstrates memory management patterns in KPU
 */

#include <sw/kpu/kpu_simulator.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

void print_memory_info(const sw::kpu::KPUSimulator& kpu) {
    std::cout << "\nMemory Configuration:\n";
    std::cout << "  Memory banks: " << kpu.get_memory_bank_count() << "\n";
    for (size_t i = 0; i < kpu.get_memory_bank_count(); ++i) {
        std::cout << "    Bank " << i << ": "
                  << (kpu.get_memory_bank_capacity(i) / (1024 * 1024)) << " MB\n";
    }

    std::cout << "  Scratchpads: " << kpu.get_scratchpad_count() << "\n";
    for (size_t i = 0; i < kpu.get_scratchpad_count(); ++i) {
        std::cout << "    Scratchpad " << i << ": "
                  << (kpu.get_scratchpad_capacity(i) / 1024) << " KB\n";
    }

    std::cout << "  L3 tiles: " << kpu.get_l3_tile_count() << "\n";
    std::cout << "  L2 banks: " << kpu.get_l2_bank_count() << "\n";
}

int main() {
    std::cout << "===========================================\n";
    std::cout << " KPU Memory Management Example\n";
    std::cout << "===========================================\n";

    // Create KPU with multiple memory banks and scratchpads
    sw::kpu::KPUSimulator::Config config(
        4,      // 4 memory banks
        512,    // 512MB each
        100,    // 100 GB/s bandwidth
        4,      // 4 scratchpads
        64,     // 64KB each
        2,      // 2 compute tiles
        4       // 4 DMA engines
    );

    sw::kpu::KPUSimulator kpu(config);
    print_memory_info(kpu);

    // Demonstrate memory operations
    std::cout << "\n===========================================\n";
    std::cout << " Memory Operations Demo\n";
    std::cout << "===========================================\n";

    // 1. Write to external memory bank
    std::cout << "\n1. Writing to external memory banks...\n";
    std::vector<float> data(1024);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    const size_t bank_id = 0;
    const size_t addr = 0;
    kpu.write_memory_bank(bank_id, addr, data.data(), data.size() * sizeof(float));
    std::cout << "  Written " << data.size() << " floats to bank " << bank_id << "\n";

    // 2. Read back from memory
    std::cout << "\n2. Reading from external memory...\n";
    std::vector<float> read_data(1024);
    kpu.read_memory_bank(bank_id, addr, read_data.data(), read_data.size() * sizeof(float));
    std::cout << "  Read " << read_data.size() << " floats from bank " << bank_id << "\n";

    // Verify
    bool match = true;
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] != read_data[i]) {
            match = false;
            break;
        }
    }
    std::cout << "  Data verification: " << (match ? "PASS" : "FAIL") << "\n";

    // 3. DMA transfer to scratchpad
    std::cout << "\n3. DMA transfer from external memory to scratchpad...\n";
    const size_t dma_id = 0;
    const size_t scratchpad_id = 0;
    const size_t transfer_size = 256 * sizeof(float);

    std::cout << "  Starting DMA transfer...\n";
    Address global_src = kpu.get_external_bank_base(bank_id) + addr;
    Address global_dst = kpu.get_scratchpad_base(scratchpad_id) + 0;
    kpu.dma_external_to_scratchpad(
        dma_id,
        global_src, global_dst,
        transfer_size
    );

    std::cout << "  DMA transfer initiated\n";
    std::cout << "  Running simulation until idle...\n";
    kpu.run_until_idle();
    std::cout << "  DMA transfer complete\n";

    // 4. Read from scratchpad
    std::cout << "\n4. Reading from scratchpad...\n";
    std::vector<float> scratch_data(256);
    kpu.read_scratchpad(scratchpad_id, 0, scratch_data.data(), scratch_data.size() * sizeof(float));
    std::cout << "  Read " << scratch_data.size() << " floats from scratchpad " << scratchpad_id << "\n";

    // Verify scratchpad data matches original
    match = true;
    for (size_t i = 0; i < scratch_data.size(); ++i) {
        if (scratch_data[i] != data[i]) {
            match = false;
            break;
        }
    }
    std::cout << "  Scratchpad data verification: " << (match ? "PASS" : "FAIL") << "\n";

    // 5. Write to scratchpad and DMA back
    std::cout << "\n5. Write to scratchpad and DMA back to memory...\n";
    std::vector<float> new_data(256);
    for (size_t i = 0; i < new_data.size(); ++i) {
        new_data[i] = static_cast<float>(i * 2);
    }

    kpu.write_scratchpad(scratchpad_id, 0, new_data.data(), new_data.size() * sizeof(float));
    std::cout << "  Written " << new_data.size() << " floats to scratchpad\n";

    // DMA back to a different memory location
    const size_t new_addr = 64 * 1024; // 64KB offset
    Address global_scratch_src = kpu.get_scratchpad_base(scratchpad_id) + 0;
    Address global_ext_dst = kpu.get_external_bank_base(bank_id) + new_addr;
    kpu.dma_scratchpad_to_external(
        dma_id,
        global_scratch_src, global_ext_dst,
        new_data.size() * sizeof(float)
    );

    kpu.run_until_idle();
    std::cout << "  DMA transfer back to memory complete\n";

    // Read and verify
    std::vector<float> final_data(256);
    kpu.read_memory_bank(bank_id, new_addr, final_data.data(), final_data.size() * sizeof(float));

    match = true;
    for (size_t i = 0; i < final_data.size(); ++i) {
        if (final_data[i] != new_data[i]) {
            match = false;
            break;
        }
    }
    std::cout << "  Final data verification: " << (match ? "PASS" : "FAIL") << "\n";

    // Print simulation statistics
    std::cout << "\n===========================================\n";
    std::cout << " Simulation Statistics\n";
    std::cout << "===========================================\n";
    std::cout << "  Total cycles: " << kpu.get_current_cycle() << "\n";
    std::cout << "  Elapsed time: " << kpu.get_elapsed_time_ms() << " ms\n";

    kpu.print_component_status();

    std::cout << "\n===========================================\n";
    std::cout << " All memory operations completed!\n";
    std::cout << "===========================================\n";

    return 0;
}
