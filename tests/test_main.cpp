#include "sw/kpu/simulator.hpp"
#include <iostream>

int main() {
    std::cout << "=== KPU Simulator Test ===" << std::endl;
    
    try {
        // Create simulator with default configuration
        sw::kpu::KPUSimulator::Config config;
        config.external_memory_mb = 256;  // 256 MB external memory
        config.scratchpad_kb = 128;       // 128 KB scratchpad
        config.memory_bandwidth_gbps = 50; // 50 Gbps memory bandwidth
        
        sw::kpu::KPUSimulator simulator(config);
        
        std::cout << "Simulator initialized successfully!" << std::endl;
        
        // Generate a simple 4x4 matrix multiplication test
        auto test = sw::kpu::test_utils::generate_simple_matmul_test(4, 4, 4);
        
        std::cout << "\nRunning 4x4 matrix multiplication test..." << std::endl;
        std::cout << "Matrix A (4x4):" << std::endl;
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                std::cout << std::fixed << std::setprecision(2) << test.matrix_a[i * 4 + j] << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "\nMatrix B (4x4):" << std::endl;
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                std::cout << std::fixed << std::setprecision(2) << test.matrix_b[i * 4 + j] << " ";
            }
            std::cout << std::endl;
        }
        
        // Run the test
        bool success = simulator.run_matmul_test(test);
        
        if (success) {
            std::cout << "\nMatrix multiplication test PASSED!" << std::endl;
            
            std::cout << "\nExpected result C:" << std::endl;
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    std::cout << std::fixed << std::setprecision(2) << test.expected_c[i * 4 + j] << " ";
                }
                std::cout << std::endl;
            }
        } else {
            std::cout << "\nMatrix multiplication test FAILED!" << std::endl;
            return 1;
        }
        
        // Print simulation statistics
        std::cout << std::endl;
        simulator.print_stats();
        
        // Test different matrix sizes
        std::cout << "\n=== Performance Tests ===" << std::endl;
        
        std::vector<std::tuple<size_t, size_t, size_t>> test_sizes = {
            {8, 8, 8},
            {16, 16, 16},
            {32, 32, 32}
        };
        
        for (auto [m, n, k] : test_sizes) {
            std::cout << "\nTesting " << m << "x" << n << " x " << k << "x" << n << " matrix multiplication..." << std::endl;
            
            auto perf_test = sw::kpu::test_utils::generate_simple_matmul_test(m, n, k);
            
            auto start_time = std::chrono::high_resolution_clock::now();
            bool perf_success = simulator.run_matmul_test(perf_test);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            if (perf_success) {
                std::cout << "  Test passed in " << duration.count() << " mu-sec" << std::endl;
                std::cout << "  Simulation cycles: " << simulator.get_current_cycle() << std::endl;
                
                // Calculate ops per second
                size_t ops = 2 * m * n * k; // 2 ops per MAC (multiply + add)
                double ops_per_cycle = static_cast<double>(ops) / simulator.get_current_cycle();
                std::cout << "  Operations per cycle: " << std::fixed << std::setprecision(2) << ops_per_cycle << std::endl;
            } else {
                std::cout << "  Test failed!" << std::endl;
            }
        }
        
        std::cout << "\n=== All tests completed successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}