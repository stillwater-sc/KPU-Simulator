/**
 * @file config_formatter_demo.cpp
 * @brief Demonstrates the new config formatting and memory map APIs
 *
 * Shows how to:
 * 1. Use operator<< for clean config output
 * 2. Access memory maps programmatically
 * 3. Get comprehensive system reports
 */

#include <sw/system/toplevel.hpp>
#include <sw/system/config_formatter.hpp>
#include <iostream>
#include <fstream>

using namespace sw::sim;

int main() {
    std::cout << "===========================================\n";
    std::cout << " Config Formatter and Memory Map Demo\n";
    std::cout << "===========================================\n\n";

    // Demo 1: Simple config output with operator<<
    std::cout << "Demo 1: Stream operator for clean config output\n";
    std::cout << "================================================\n";

    auto config = SystemConfig::create_minimal_kpu();
    std::cout << config;  // Clean, one-line call!

    // Demo 2: Initialize simulator and get memory map
    std::cout << "\nDemo 2: Memory Map Reporting\n";
    std::cout << "==============================\n";

    SystemSimulator sim(config);
    if (sim.initialize()) {
        std::cout << "Simulator initialized successfully!\n";

        // Get memory map for KPU 0
        std::string memory_map = sim.get_memory_map(0);
        std::cout << memory_map;
    }

    // Demo 3: Comprehensive system report (config + memory maps + runtime state)
    std::cout << "\nDemo 3: Complete System Report\n";
    std::cout << "==============================\n";

    std::string full_report = sim.get_system_report();
    // Note: This includes everything - config, runtime stats, and memory maps
    std::cout << "Full report size: " << full_report.size() << " characters\n";
    std::cout << "Contains memory map: "
              << (full_report.find("Memory Map") != std::string::npos ? "Yes" : "No") << "\n";

    // Demo 4: Print to file
    std::cout << "\nDemo 4: Output to File\n";
    std::cout << "======================\n";

    std::ofstream file("system_report.txt");
    if (file.is_open()) {
        sim.print_full_report(file);
        file.close();
        std::cout << "Full system report written to: system_report.txt\n";
    }

    // Demo 5: Compare before/after
    std::cout << "\nDemo 5: The Old Way vs The New Way\n";
    std::cout << "===================================\n";

    std::cout << "\nOLD WAY (error-prone, manual):\n";
    std::cout << "-------------------------------\n";
    std::cout << "  std::cout << \"  System: \" << config.system.name << \"\\n\";\n";
    std::cout << "  std::cout << \"  Memory banks: \" \n";
    std::cout << "            << config.accelerators[0].kpu_config->memory.banks.size() << \"\\n\";\n";
    std::cout << "  // ... 10+ more lines of manual formatting ...\n";

    std::cout << "\nNEW WAY (clean, automatic):\n";
    std::cout << "----------------------------\n";
    std::cout << "  std::cout << config;  // Done!\n";
    std::cout << "  std::cout << sim.get_memory_map();  // Memory map included!\n";

    sim.shutdown();

    std::cout << "\n===========================================\n";
    std::cout << " Demo completed successfully!\n";
    std::cout << "===========================================\n";

    return 0;
}
