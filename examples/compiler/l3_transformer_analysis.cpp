/**
 * @file l3_transformer_analysis.cpp
 * @brief L3 Overfetch Analysis for Realistic Transformer Workloads
 *
 * Transformers use very large matrices (32k × 7k and beyond).
 * This tool analyzes L3 cache requirements for these workloads.
 *
 * Typical Transformer Sizes:
 * - Attention: Q×K^T where Q,K are [batch_size×seq_len, d_model]
 * - MLP layers: [batch_size×seq_len, d_model] × [d_model, 4×d_model]
 * - Large models: d_model = 7168 (7k), seq_len up to 32k tokens
 *
 * Example: LLaMA-2 70B
 * - d_model = 8192
 * - Hidden dim = 28672
 * - Matrix multiply: [32768, 8192] × [8192, 28672]
 */

#include <sw/compiler/l3_scheduler.hpp>
#include <sw/compiler/l2_tile_scheduler.hpp>
#include <sw/compiler/tile_optimizer.hpp>
#include <sw/compiler/schedule_characterizer.hpp>
#include <sw/concepts.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <chrono>

using namespace sw::kpu::compiler;
using sw::kpu::Cycle;
using sw::kpu::Size;

/**
 * @brief Print transformer workload analysis summary
 */
void print_transformer_summary(
    const TensorShape& shape,
    const L3Schedule& l3_schedule,
    const std::string& workload_name)
{
    std::cout << "\n" << workload_name << ": " << shape.to_string() << "\n";
    std::cout << "─────────────────────────────────────────────────────────────────\n";

    Size total_ideal = l3_schedule.tensor_a_size + l3_schedule.tensor_b_size + l3_schedule.tensor_c_size;

    std::cout << "  Tensor sizes:\n";
    std::cout << "    A: " << std::setw(8) << (l3_schedule.tensor_a_size / (1024 * 1024)) << " MB\n";
    std::cout << "    B: " << std::setw(8) << (l3_schedule.tensor_b_size / (1024 * 1024)) << " MB\n";
    std::cout << "    C: " << std::setw(8) << (l3_schedule.tensor_c_size / (1024 * 1024)) << " MB\n";
    std::cout << "    Total: " << std::setw(5) << (total_ideal / (1024 * 1024)) << " MB\n";

    std::cout << "  With L3=" << (l3_schedule.config.l3_capacity / (1024 * 1024)) << "MB:\n";
    std::cout << "    Overfetch: " << std::fixed << std::setprecision(2)
              << l3_schedule.overfetch_factor_total << "×\n";
    std::cout << "    Hit rate:  " << std::fixed << std::setprecision(1)
              << (l3_schedule.l3_hit_rate * 100.0) << "%\n";
    std::cout << "    DRAM reads: " << (l3_schedule.total_dram_reads / (1024 * 1024)) << " MB\n";
    std::cout << "    L3 util:   " << std::fixed << std::setprecision(1)
              << (l3_schedule.l3_utilization * 100.0) << "%\n";
}

int main() {
    std::cout << "████████████████████████████████████████████████████████████████████████████\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█         L3 OVERFETCH ANALYSIS FOR TRANSFORMER WORKLOADS                  █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "█  Realistic sizes from LLaMA, GPT, BERT, and other transformer models    █\n";
    std::cout << "█                                                                          █\n";
    std::cout << "████████████████████████████████████████████████████████████████████████████\n\n";

    // Setup
    TileOptimizer::MemoryHierarchy mem;
    TileOptimizer optimizer(mem);
    L2TileScheduler l2_scheduler;

    // Realistic transformer workloads
    std::vector<std::pair<std::string, TensorShape>> transformer_workloads = {
        // Small transformer (BERT-base style)
        {"BERT-Base Attention Q×K^T",     TensorShape(512, 512, 768)},
        {"BERT-Base MLP",                 TensorShape(512, 768, 3072)},

        // Medium transformer (GPT-2 style)
        {"GPT-2 Attention Q×K^T",         TensorShape(1024, 1024, 1536)},
        {"GPT-2 MLP",                     TensorShape(1024, 1536, 6144)},

        // Large transformer (LLaMA-2 7B style)
        {"LLaMA-2 7B Attention Q×K^T",    TensorShape(4096, 4096, 4096)},
        {"LLaMA-2 7B MLP",                TensorShape(4096, 4096, 11008)},

        // Very large transformer (LLaMA-2 70B style)
        {"LLaMA-2 70B Attention Q×K^T",   TensorShape(8192, 8192, 8192)},
        {"LLaMA-2 70B MLP",               TensorShape(8192, 8192, 28672)},

        // Extreme: Long context (32k tokens)
        {"LLaMA-2 70B Long Context Q×K^T", TensorShape(32768, 32768, 8192)},
        {"LLaMA-2 70B Long Context MLP",   TensorShape(32768, 8192, 28672)},

        // The realistic transformer size mentioned by user
        {"User Example: 32k × 7k",        TensorShape(32768, 7168, 7168)},
    };

    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       DEMO 1: Transformer Workloads with Various L3 Sizes          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";

    // Test with different L3 sizes
    std::vector<Size> l3_sizes_to_test = {
        16 * 1024 * 1024,   // 16MB
        64 * 1024 * 1024,   // 64MB
        256 * 1024 * 1024   // 256MB
    };

    for (Size l3_size : l3_sizes_to_test) {
        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "Testing with L3 = " << (l3_size / (1024 * 1024)) << " MB\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

        L3Config l3_config;
        l3_config.l3_capacity = l3_size;
        L3Scheduler l3_scheduler(l3_config);

        for (const auto& [name, shape] : transformer_workloads) {
            auto tile_config = optimizer.optimize(shape.M, shape.N, shape.K);
            auto l2_schedule = l2_scheduler.generate_schedule(shape.M, shape.N, shape.K, tile_config);
            auto l3_schedule = l3_scheduler.schedule_l3(shape, l2_schedule);

            print_transformer_summary(shape, l3_schedule, name);
        }
    }

    // Demo 2: L3 sweep for one large workload
    std::cout << "\n\n╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       DEMO 2: L3 Size Sweep for Large Transformer Workload        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    TensorShape large_transformer(32768, 7168, 7168);  // User's example
    std::cout << "Analyzing: " << large_transformer.to_string() << "\n";

    Size total_tensor_size = large_transformer.M * large_transformer.K +
                            large_transformer.K * large_transformer.N +
                            large_transformer.M * large_transformer.N;
    total_tensor_size *= 4; // FP32
    std::cout << "Total tensor size: " << (total_tensor_size / (1024 * 1024)) << " MB\n\n";

    // Create L2 schedule
    auto tile_config = optimizer.optimize(large_transformer.M, large_transformer.N, large_transformer.K);
    auto l2_schedule = l2_scheduler.generate_schedule(
        large_transformer.M, large_transformer.N, large_transformer.K, tile_config);

    // Sweep L3 sizes
    std::vector<Size> l3_sweep_sizes = {
        8 * 1024 * 1024,     // 8MB
        16 * 1024 * 1024,    // 16MB
        32 * 1024 * 1024,    // 32MB
        64 * 1024 * 1024,    // 64MB
        128 * 1024 * 1024,   // 128MB
        256 * 1024 * 1024,   // 256MB
        512 * 1024 * 1024,   // 512MB
        1024 * 1024 * 1024   // 1GB
    };

    L3Config base_config;
    L3Scheduler sweep_scheduler(base_config);

    auto sweep_results = sweep_scheduler.sweep_l3_size(large_transformer, l2_schedule, l3_sweep_sizes);

    std::cout << "┌──────────┬──────────────┬──────────────┬──────────┬──────────┐\n";
    std::cout << "│ L3 Size  │  Overfetch   │  DRAM Reads  │ L3 Hit   │  L3 Util │\n";
    std::cout << "│   (MB)   │    Factor    │     (MB)     │  Rate    │   (%)    │\n";
    std::cout << "├──────────┼──────────────┼──────────────┼──────────┼──────────┤\n";

    for (const auto& [l3_size, schedule] : sweep_results) {
        std::cout << "│ " << std::setw(8) << (l3_size / (1024 * 1024))
                  << " │ " << std::setw(12) << std::fixed << std::setprecision(2)
                  << schedule.overfetch_factor_total
                  << " │ " << std::setw(12) << (schedule.total_dram_reads / (1024 * 1024))
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(1)
                  << (schedule.l3_hit_rate * 100.0)
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(1)
                  << (schedule.l3_utilization * 100.0)
                  << " │\n";
    }

    std::cout << "└──────────┴──────────────┴──────────────┴──────────┴──────────┘\n\n";

    // Export the sweep data
    std::ofstream csv("l3_overfetch_transformer_32kx7k.csv");
    csv << "L3_Size_MB,Overfetch_Total,Overfetch_A,Overfetch_B,Overfetch_C,"
        << "DRAM_Reads_KB,DRAM_Writes_KB,L3_Hit_Rate,L3_Utilization,Num_Evictions,"
        << "Peak_L3_KB,Tensor_A_KB,Tensor_B_KB,Tensor_C_KB\n";

    for (const auto& [l3_size, schedule] : sweep_results) {
        csv << (l3_size / (1024 * 1024)) << ","
            << schedule.overfetch_factor_total << ","
            << schedule.overfetch_factor_a << ","
            << schedule.overfetch_factor_b << ","
            << schedule.overfetch_factor_c << ","
            << (schedule.total_dram_reads / 1024) << ","
            << (schedule.total_dram_writes / 1024) << ","
            << schedule.l3_hit_rate << ","
            << schedule.l3_utilization << ","
            << schedule.num_l3_evictions << ","
            << (schedule.l3_capacity_used_peak / 1024) << ","
            << (schedule.tensor_a_size / 1024) << ","
            << (schedule.tensor_b_size / 1024) << ","
            << (schedule.tensor_c_size / 1024) << "\n";
    }
    csv.close();

    std::cout << "Exported sweep to l3_overfetch_transformer_32kx7k.csv\n\n";

    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    ANALYSIS COMPLETE                               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Key Findings for Large Transformers:\n";
    std::cout << "  1. Tensor sizes are MUCH larger (100s of MB vs KB)\n";
    std::cout << "  2. L3 cache needs to be correspondingly larger (64-256MB)\n";
    std::cout << "  3. Overfetch still shows diminishing returns curve\n";
    std::cout << "  4. Hit rate stabilizes at a certain L3 size (the knee)\n\n";

    std::cout << "Visualization:\n";
    std::cout << "  python ../tools/compiler/visualize_l3_overfetch.py l3_overfetch_transformer_32kx7k.csv\n\n";

    return 0;
}
