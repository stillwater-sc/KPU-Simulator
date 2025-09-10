#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "sw/kpu/kpu_simulator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(stillwater_kpu, m) {
    m.doc() = "Stillwater KPU Simulator - High-performance C++ KPU simulator with Python bindings";
    
    // Version information
    m.attr("__version__") = PYBIND11_STRINGIFY(VERSION_INFO);
    
    // Basic types
    py::class_<sw::kpu::ExternalMemory>(m, "ExternalMemory")
        .def("get_capacity", &sw::kpu::ExternalMemory::get_capacity)
        .def("get_bandwidth", &sw::kpu::ExternalMemory::get_bandwidth)
        .def("is_ready", &sw::kpu::ExternalMemory::is_ready)
        .def("reset", &sw::kpu::ExternalMemory::reset)
        .def("get_last_access_cycle", &sw::kpu::ExternalMemory::get_last_access_cycle);
    
    py::class_<sw::kpu::Scratchpad>(m, "Scratchpad")
        .def("get_capacity", &sw::kpu::Scratchpad::get_capacity)
        .def("is_ready", &sw::kpu::Scratchpad::is_ready)
        .def("reset", &sw::kpu::Scratchpad::reset);
    
    py::enum_<sw::kpu::DMAEngine::MemoryType>(m, "MemoryType")
        .value("EXTERNAL", sw::kpu::DMAEngine::MemoryType::EXTERNAL)
        .value("SCRATCHPAD", sw::kpu::DMAEngine::MemoryType::SCRATCHPAD);
    
    py::class_<sw::kpu::DMAEngine>(m, "DMAEngine")
        .def("is_busy", &sw::kpu::DMAEngine::is_busy)
        .def("reset", &sw::kpu::DMAEngine::reset)
        .def("get_src_type", &sw::kpu::DMAEngine::get_src_type)
        .def("get_dst_type", &sw::kpu::DMAEngine::get_dst_type)
        .def("get_src_id", &sw::kpu::DMAEngine::get_src_id)
        .def("get_dst_id", &sw::kpu::DMAEngine::get_dst_id);
    
    py::class_<sw::kpu::ComputeFabric>(m, "ComputeFabric")
        .def("is_busy", &sw::kpu::ComputeFabric::is_busy)
        .def("reset", &sw::kpu::ComputeFabric::reset)
        .def("get_tile_id", &sw::kpu::ComputeFabric::get_tile_id);
    
    py::class_<sw::kpu::KPUSimulator::Config>(m, "SimulatorConfig")
        .def(py::init<>())
        .def_readwrite("memory_bank_count", &sw::kpu::KPUSimulator::Config::memory_bank_count)
        .def_readwrite("memory_bank_capacity_mb", &sw::kpu::KPUSimulator::Config::memory_bank_capacity_mb)
        .def_readwrite("memory_bandwidth_gbps", &sw::kpu::KPUSimulator::Config::memory_bandwidth_gbps)
        .def_readwrite("scratchpad_count", &sw::kpu::KPUSimulator::Config::scratchpad_count)
        .def_readwrite("scratchpad_capacity_kb", &sw::kpu::KPUSimulator::Config::scratchpad_capacity_kb)
        .def_readwrite("compute_tile_count", &sw::kpu::KPUSimulator::Config::compute_tile_count)
        .def_readwrite("dma_engine_count", &sw::kpu::KPUSimulator::Config::dma_engine_count);
    
    py::class_<sw::kpu::KPUSimulator::MatMulTest>(m, "MatMulTest")
        .def(py::init<>())
        .def_readwrite("m", &sw::kpu::KPUSimulator::MatMulTest::m)
        .def_readwrite("n", &sw::kpu::KPUSimulator::MatMulTest::n)
        .def_readwrite("k", &sw::kpu::KPUSimulator::MatMulTest::k)
        .def_readwrite("matrix_a", &sw::kpu::KPUSimulator::MatMulTest::matrix_a)
        .def_readwrite("matrix_b", &sw::kpu::KPUSimulator::MatMulTest::matrix_b)
        .def_readwrite("expected_c", &sw::kpu::KPUSimulator::MatMulTest::expected_c);
    
    py::class_<sw::kpu::KPUSimulator>(m, "KPUSimulator")
        .def(py::init<const sw::kpu::KPUSimulator::Config&>(), py::arg("config") = sw::kpu::KPUSimulator::Config{})
        
        // Memory operations - clean delegation API
        .def("read_memory_bank", [](sw::kpu::KPUSimulator& self, size_t bank_id, sw::kpu::Address addr, size_t count) {
            std::vector<float> data(count);
            self.read_memory_bank(bank_id, addr, data.data(), count * sizeof(float));
            return data;
        })
        .def("write_memory_bank", [](sw::kpu::KPUSimulator& self, size_t bank_id, sw::kpu::Address addr, const std::vector<float>& data) {
            self.write_memory_bank(bank_id, addr, data.data(), data.size() * sizeof(float));
        })
        .def("read_scratchpad", [](sw::kpu::KPUSimulator& self, size_t pad_id, sw::kpu::Address addr, size_t count) {
            std::vector<float> data(count);
            self.read_scratchpad(pad_id, addr, data.data(), count * sizeof(float));
            return data;
        })
        .def("write_scratchpad", [](sw::kpu::KPUSimulator& self, size_t pad_id, sw::kpu::Address addr, const std::vector<float>& data) {
            self.write_scratchpad(pad_id, addr, data.data(), data.size() * sizeof(float));
        })
        
        // NumPy array support
        .def("read_memory_bank_numpy", [](sw::kpu::KPUSimulator& self, size_t bank_id, sw::kpu::Address addr, const std::vector<size_t>& shape) {
            size_t total_elements = 1;
            for (auto dim : shape) total_elements *= dim;
            
            auto result = py::array_t<float>(shape);
            self.read_memory_bank(bank_id, addr, result.mutable_unchecked().mutable_data(0), total_elements * sizeof(float));
            return result;
        })
        .def("write_memory_bank_numpy", [](sw::kpu::KPUSimulator& self, size_t bank_id, sw::kpu::Address addr, py::array_t<float> array) {
            py::buffer_info buf = array.request();
            self.write_memory_bank(bank_id, addr, buf.ptr, buf.size * sizeof(float));
        })
        .def("read_scratchpad_numpy", [](sw::kpu::KPUSimulator& self, size_t pad_id, sw::kpu::Address addr, const std::vector<size_t>& shape) {
            size_t total_elements = 1;
            for (auto dim : shape) total_elements *= dim;
            
            auto result = py::array_t<float>(shape);
            self.read_scratchpad(pad_id, addr, result.mutable_unchecked().mutable_data(0), total_elements * sizeof(float));
            return result;
        })
        .def("write_scratchpad_numpy", [](sw::kpu::KPUSimulator& self, size_t pad_id, sw::kpu::Address addr, py::array_t<float> array) {
            py::buffer_info buf = array.request();
            self.write_scratchpad(pad_id, addr, buf.ptr, buf.size * sizeof(float));
        })
        
        // DMA operations
        .def("start_dma_transfer", [](sw::kpu::KPUSimulator& self, size_t dma_id, sw::kpu::Address src_addr, sw::kpu::Address dst_addr, 
                                     sw::kpu::Size size, py::function callback) {
            if (callback.is_none()) {
                self.start_dma_transfer(dma_id, src_addr, dst_addr, size);
            } else {
                self.start_dma_transfer(dma_id, src_addr, dst_addr, size, [callback]() { callback(); });
            }
        }, py::arg("dma_id"), py::arg("src_addr"), py::arg("dst_addr"), py::arg("size"), py::arg("callback") = py::none())
        .def("is_dma_busy", &sw::kpu::KPUSimulator::is_dma_busy)
        
        // Compute operations
        .def("start_matmul", [](sw::kpu::KPUSimulator& self, size_t tile_id, size_t scratchpad_id, sw::kpu::Size m, sw::kpu::Size n, sw::kpu::Size k,
                               sw::kpu::Address a_addr, sw::kpu::Address b_addr, sw::kpu::Address c_addr, py::function callback) {
            if (callback.is_none()) {
                self.start_matmul(tile_id, scratchpad_id, m, n, k, a_addr, b_addr, c_addr);
            } else {
                self.start_matmul(tile_id, scratchpad_id, m, n, k, a_addr, b_addr, c_addr, [callback]() { callback(); });
            }
        }, py::arg("tile_id"), py::arg("scratchpad_id"), py::arg("m"), py::arg("n"), py::arg("k"), 
           py::arg("a_addr"), py::arg("b_addr"), py::arg("c_addr"), py::arg("callback") = py::none())
        .def("is_compute_busy", &sw::kpu::KPUSimulator::is_compute_busy)
        
        // Simulation control
        .def("reset", &sw::kpu::KPUSimulator::reset)
        .def("step", &sw::kpu::KPUSimulator::step)
        .def("run_until_idle", &sw::kpu::KPUSimulator::run_until_idle)
        
        // Configuration queries
        .def("get_memory_bank_count", &sw::kpu::KPUSimulator::get_memory_bank_count)
        .def("get_scratchpad_count", &sw::kpu::KPUSimulator::get_scratchpad_count)
        .def("get_compute_tile_count", &sw::kpu::KPUSimulator::get_compute_tile_count)
        .def("get_dma_engine_count", &sw::kpu::KPUSimulator::get_dma_engine_count)
        .def("get_memory_bank_capacity", &sw::kpu::KPUSimulator::get_memory_bank_capacity)
        .def("get_scratchpad_capacity", &sw::kpu::KPUSimulator::get_scratchpad_capacity)
        
        // High-level operations
        .def("run_matmul_test", &sw::kpu::KPUSimulator::run_matmul_test,
             py::arg("test"), py::arg("memory_bank_id") = 0, py::arg("scratchpad_id") = 0, py::arg("compute_tile_id") = 0)
        
        // Statistics and monitoring
        .def("get_current_cycle", &sw::kpu::KPUSimulator::get_current_cycle)
        .def("get_elapsed_time_ms", &sw::kpu::KPUSimulator::get_elapsed_time_ms)
        .def("print_stats", &sw::kpu::KPUSimulator::print_stats)
        .def("print_component_status", &sw::kpu::KPUSimulator::print_component_status)
        .def("is_memory_bank_ready", &sw::kpu::KPUSimulator::is_memory_bank_ready)
        .def("is_scratchpad_ready", &sw::kpu::KPUSimulator::is_scratchpad_ready)
        
        // Convenient numpy matrix multiplication
        .def("run_numpy_matmul", [](sw::kpu::KPUSimulator& self, py::array_t<float> a, py::array_t<float> b,
                                   size_t memory_bank_id, size_t scratchpad_id, size_t compute_tile_id) {
            py::buffer_info a_buf = a.request();
            py::buffer_info b_buf = b.request();
            
            if (a_buf.ndim != 2 || b_buf.ndim != 2) {
                throw std::runtime_error("Input arrays must be 2-dimensional");
            }
            
            sw::kpu::Size m = a_buf.shape[0];
            sw::kpu::Size k = a_buf.shape[1];
            sw::kpu::Size n = b_buf.shape[1];
            
            if (k != static_cast<sw::kpu::Size>(b_buf.shape[0])) {
                throw std::runtime_error("Matrix dimensions don't match for multiplication");
            }
            
            // Create test structure
            sw::kpu::KPUSimulator::MatMulTest test;
            test.m = m;
            test.n = n;
            test.k = k;
            
            // Copy data from numpy arrays
            test.matrix_a.assign(static_cast<float*>(a_buf.ptr), 
                               static_cast<float*>(a_buf.ptr) + a_buf.size);
            test.matrix_b.assign(static_cast<float*>(b_buf.ptr), 
                               static_cast<float*>(b_buf.ptr) + b_buf.size);
            
            // Compute expected result
            test.expected_c.resize(m * n);
            for (sw::kpu::Size i = 0; i < m; ++i) {
                for (sw::kpu::Size j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (sw::kpu::Size ki = 0; ki < k; ++ki) {
                        sum += test.matrix_a[i * k + ki] * test.matrix_b[ki * n + j];
                    }
                    test.expected_c[i * n + j] = sum;
                }
            }
            
            // Run simulation
            bool success = self.run_matmul_test(test, memory_bank_id, scratchpad_id, compute_tile_id);
            
            if (!success) {
                throw std::runtime_error("Matrix multiplication simulation failed");
            }
            
            // Return result as numpy array
            auto result = py::array_t<float>({m, n});
            py::buffer_info result_buf = result.request();
            std::copy(test.expected_c.begin(), test.expected_c.end(), static_cast<float*>(result_buf.ptr));
            
            return result;
        }, py::arg("a"), py::arg("b"), py::arg("memory_bank_id") = 0, py::arg("scratchpad_id") = 0, py::arg("compute_tile_id") = 0);
    
    // Test utilities
    m.def("generate_simple_matmul_test", &sw::kpu::test_utils::generate_simple_matmul_test,
          py::arg("m") = 4, py::arg("n") = 4, py::arg("k") = 4);
    
    m.def("generate_random_matrix", &sw::kpu::test_utils::generate_random_matrix,
          py::arg("rows"), py::arg("cols"), py::arg("min_val") = -1.0f, py::arg("max_val") = 1.0f);
    
    m.def("verify_matmul_result", &sw::kpu::test_utils::verify_matmul_result,
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("m"), py::arg("n"), py::arg("k"), 
          py::arg("tolerance") = 1e-5f);
    
    m.def("generate_multi_bank_config", &sw::kpu::test_utils::generate_multi_bank_config,
          py::arg("num_banks") = 4, py::arg("num_tiles") = 2);
    
    m.def("run_distributed_matmul_test", &sw::kpu::test_utils::run_distributed_matmul_test,
          py::arg("sim"), py::arg("matrix_size") = 8);
    
    m.def("generate_simple_matmul_test", &sw::kpu::test_utils::generate_simple_matmul_test,
          py::arg("m") = 4, py::arg("n") = 4, py::arg("k") = 4);
    
    m.def("generate_random_matrix", &sw::kpu::test_utils::generate_random_matrix,
          py::arg("rows"), py::arg("cols"), py::arg("min_val") = -1.0f, py::arg("max_val") = 1.0f);
    
    m.def("verify_matmul_result", &sw::kpu::test_utils::verify_matmul_result,
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("m"), py::arg("n"), py::arg("k"), 
          py::arg("tolerance") = 1e-5f);
    
    // Convenience function for numpy integration
    //m.def("numpy_matmul", [](py::array_t<float> a, py::array_t<float> b, const sw::kpu::KPUSimulator::Config& config) {
    //    sw::kpu::KPUSimulator simulator(config);
    //    return simulator.run_numpy_matmul(a, b);
    //}, py::arg("a"), py::arg("b"), py::arg("config") = sw::kpu::KPUSimulator::Config{});
}