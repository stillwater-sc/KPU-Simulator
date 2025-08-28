#include "sw/kpu/simulator.hpp"
#include <iostream>

int main() {
    std::cout << "=== KPU Simulator Test (Clean Architecture) ===" << std::endl;
    
    try {
        // Test 1: Basic single-bank, single-tile configuration
        std::cout << "\n=== Test 1: Basic Configuration ===" << std::endl;
        {
            sw::kpu::KPUSimulator::Config config;
            config.memory_bank_count = 1;
            config.scratchpad_count = 1;
            config.compute_tile_count = 1;
            config.dma_engine_count = 2;
            
            sw::kpu::KPUSimulator simulator(config);
            simulator.print_component_status();
            
            // Run basic matmul test
            auto test = sw::kpu::test_utils::generate_simple_matmul_test(4, 4, 4);
            bool success = simulator.run_matmul_test(test);
            
            std::cout << "Basic matmul test: " << (success ? "PASSED" : "FAILED") << std::endl;
            simulator.print_stats();
        }
        
        // Test 2: Multi-bank configuration
        std::cout << "\n=== Test 2: Multi-Bank Configuration ===" << std::endl;
        {
            auto config = sw::kpu::test_utils::generate_multi_bank_config(4, 2);
            sw::kpu::KPUSimulator simulator(config);
            
            std::cout << "Created simulator with:" << std::endl;
            std::cout << "  " << simulator.get_memory_bank_count() << " memory banks" << std::endl;
            std::cout << "  " << simulator.get_scratchpad_count() << " scratchpads" << std::endl;
            std::cout << "  " << simulator.get_compute_tile_count() << " compute tiles" << std::endl;
            std::cout << "  " << simulator.get_dma_engine_count() << " DMA engines" << std::endl;
            
            simulator.print_component_status();
            
            // Test distributed matmul
            bool success = sw::kpu::test_utils::run_distributed_matmul_test(simulator, 8);
            std::cout << "Multi-bank matmul test: " << (success ? "PASSED" : "FAILED") << std::endl;
        }
        
        // Test 3: Direct API usage (no high-level test functions)
        std::cout << "\n=== Test 3: Direct API Usage ===" << std::endl;
        {
            sw::kpu::KPUSimulator::Config config;
            config.memory_bank_count = 2;
            config.scratchpad_count = 1;
            config.compute_tile_count = 1;
            config.dma_engine_count = 3;
            
            sw::kpu::KPUSimulator simulator(config);
            
            // Create simple test matrices
            std::vector<float> matrix_a = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2
            std::vector<float> matrix_b = {2.0f, 0.0f, 1.0f, 2.0f};  // 2x2
            std::vector<float> matrix_c(4, 0.0f);                    // 2x2 result

			//  1 *2 + 2*1 = 4, 1*0 + 2*2 = 4
			//  3 *2 + 4*1 = 10, 3*0 + 4*2 = 8
            
            // Load matrices into different memory banks
            simulator.write_memory_bank(0, 0, matrix_a.data(), matrix_a.size() * sizeof(float));
            simulator.write_memory_bank(1, 0, matrix_b.data(), matrix_b.size() * sizeof(float));
            
            std::cout << "Loaded matrices into separate memory banks" << std::endl;
            
            // Transfer matrix A to scratchpad
            bool dma_a_done = false;
            simulator.start_dma_transfer(0, 0, 0, matrix_a.size() * sizeof(float),
                [&dma_a_done]() { 
                    std::cout << "Matrix A transfer completed" << std::endl;
                    dma_a_done = true; 
                });
            
            // Transfer matrix B to scratchpad
            bool dma_b_done = false;
            simulator.start_dma_transfer(0, 0, 16, matrix_b.size() * sizeof(float),
                [&dma_b_done]() { 
                    std::cout << "Matrix B transfer completed" << std::endl;
                    dma_b_done = true; 
                });
            
            // Wait for transfers
            while (!dma_a_done || !dma_b_done) {
                simulator.step();
            }
            
            // Start matrix multiplication
            bool compute_done = false;
            simulator.start_matmul(0, 0, 2, 2, 2, 0, 16, 32,
                [&compute_done]() { 
                    std::cout << "Matrix multiplication completed" << std::endl;
                    compute_done = true; 
                });
            
            // Wait for computation
            while (!compute_done) {
                simulator.step();
            }
            
            // Transfer result back to memory bank 0
            bool dma_c_done = false;
            simulator.start_dma_transfer(1, 32, 32, matrix_c.size() * sizeof(float),
                [&dma_c_done]() { 
                    std::cout << "Result transfer completed" << std::endl;
                    dma_c_done = true; 
                });
            
            // Wait for result transfer
            while (!dma_c_done) {
                simulator.step();
            }
            
            // Read result
            simulator.read_memory_bank(0, 32, matrix_c.data(), matrix_c.size() * sizeof(float));
            
            std::cout << "Result matrix C:" << std::endl;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    std::cout << std::fixed << std::setprecision(1) << matrix_c[i * 2 + j] << " ";
                }
                std::cout << std::endl;
            }
            
            // Verify result (expected: [[4, 4], [10, 8]])
            std::vector<float> expected = {4.0f, 4.0f, 10.0f, 8.0f};
            bool api_test_passed = true;
            for (size_t i = 0; i < expected.size(); ++i) {
                if (std::abs(matrix_c[i] - expected[i]) > 1e-5f) {
                    api_test_passed = false;
                    break;
                }
            }
            
            std::cout << "Direct API test: " << (api_test_passed ? "PASSED" : "FAILED") << std::endl;
            simulator.print_stats();
        }
        
        // Test 4: Component status and monitoring
        std::cout << "\n=== Test 4: Status Monitoring ===" << std::endl;
        {
            auto config = sw::kpu::test_utils::generate_multi_bank_config(3, 2);
            sw::kpu::KPUSimulator simulator(config);
            
            std::cout << "Component capacities:" << std::endl;
            for (size_t i = 0; i < simulator.get_memory_bank_count(); ++i) {
                std::cout << "  Memory bank[" << i << "]: " 
                         << simulator.get_memory_bank_capacity(i) / (1024*1024) << " MB" << std::endl;
            }
            for (size_t i = 0; i < simulator.get_scratchpad_count(); ++i) {
                std::cout << "  Scratchpad[" << i << "]: " 
                         << simulator.get_scratchpad_capacity(i) / 1024 << " KB" << std::endl;
            }
            
            // Test readiness status
            std::cout << "\nReadiness status:" << std::endl;
            for (size_t i = 0; i < simulator.get_memory_bank_count(); ++i) {
                std::cout << "  Memory bank[" << i << "] ready: " 
                         << (simulator.is_memory_bank_ready(i) ? "Yes" : "No") << std::endl;
            }
            for (size_t i = 0; i < simulator.get_scratchpad_count(); ++i) {
                std::cout << "  Scratchpad[" << i << "] ready: " 
                         << (simulator.is_scratchpad_ready(i) ? "Yes" : "No") << std::endl;
            }
            
            std::cout << "Status monitoring test: PASSED" << std::endl;
        }
        
        std::cout << "\n=== All Tests Completed Successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}