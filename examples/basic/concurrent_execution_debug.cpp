/**
 * @file concurrent_execution_debug.cpp
 * @brief Debug example for concurrent resource execution
 *
 * This standalone example is extracted for debugging the concurrent
 * execution model. The current output shows non-sensical schedules:
 *
 * Issues observed:
 * 1. BM[2], BM[3], STR[2], STR[3] are completely idle (0% utilization)
 * 2. DMA operations show only B and C, rarely A in the timeline
 * 3. The cycle-by-cycle view shows only DMA0 active for first 30 cycles
 * 4. Barriers don't seem to properly sequence operations
 */

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/kpu/isa/concurrent_executor.hpp>
#include <iostream>
#include <iomanip>

using namespace sw::kpu::isa;
using sw::kpu::Size;
using sw::kpu::Cycle;

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(80, '=') << "\n";
}

// ============================================================================
// Debug: Trace instruction scheduling
// ============================================================================

void debug_instruction_scheduling() {
    print_separator("Debug: Instruction-Level Scheduling Analysis");

    // Use a tiny matmul for detailed analysis
    OutputStationaryProgramBuilder::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 32;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.L1_Ki = 16;
    config.systolic_size = 16;
    config.element_size = 4;

    config.l3_tile_capacity = 128 * 1024;
    config.l2_bank_capacity = 64 * 1024;
    config.l1_buffer_capacity = 32 * 1024;
    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.num_l1_buffers = 4;
    config.double_buffer = false;

    OutputStationaryProgramBuilder builder(config);
    DMProgram program = builder.build();

    std::cout << "\nProgram: " << program.name << "\n";
    std::cout << "Total instructions: " << program.instructions.size() << "\n\n";

    // Print all instructions with their details
    std::cout << "Full Instruction Listing:\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::setw(4) << "PC" << " | "
              << std::setw(20) << "Opcode" << " | "
              << std::setw(6) << "Matrix" << " | "
              << std::setw(12) << "Tile[ti,tj,tk]" << " | "
              << std::setw(10) << "Size" << " | "
              << "Label\n";
    std::cout << std::string(100, '-') << "\n";

    for (size_t i = 0; i < program.instructions.size(); ++i) {
        const auto& instr = program.instructions[i];

        std::string opcode_str;
        std::string matrix_str = "-";
        std::string tile_str = "-";
        std::string size_str = "-";

        switch (instr.opcode) {
            case DMOpcode::DMA_LOAD_TILE: opcode_str = "DMA_LOAD_TILE"; break;
            case DMOpcode::DMA_STORE_TILE: opcode_str = "DMA_STORE_TILE"; break;
            case DMOpcode::BM_MOVE_TILE: opcode_str = "BM_MOVE_TILE"; break;
            case DMOpcode::BM_TRANSPOSE_TILE: opcode_str = "BM_TRANSPOSE_TILE"; break;
            case DMOpcode::STR_FEED_ROWS: opcode_str = "STR_FEED_ROWS"; break;
            case DMOpcode::STR_FEED_COLS: opcode_str = "STR_FEED_COLS"; break;
            case DMOpcode::STR_DRAIN_OUTPUT: opcode_str = "STR_DRAIN_OUTPUT"; break;
            case DMOpcode::BARRIER: opcode_str = "BARRIER"; break;
            case DMOpcode::HALT: opcode_str = "HALT"; break;
            default: opcode_str = "OTHER"; break;
        }

        // Extract operand details
        if (std::holds_alternative<DMAOperands>(instr.operands)) {
            const auto& ops = std::get<DMAOperands>(instr.operands);
            matrix_str = (ops.matrix == MatrixID::A) ? "A" :
                        (ops.matrix == MatrixID::B) ? "B" : "C";
            tile_str = "[" + std::to_string(ops.tile.ti) + "," +
                      std::to_string(ops.tile.tj) + "," +
                      std::to_string(ops.tile.tk) + "]";
            size_str = std::to_string(ops.size_bytes);
        } else if (std::holds_alternative<BlockMoverOperands>(instr.operands)) {
            const auto& ops = std::get<BlockMoverOperands>(instr.operands);
            matrix_str = (ops.matrix == MatrixID::A) ? "A" :
                        (ops.matrix == MatrixID::B) ? "B" : "C";
            tile_str = "[" + std::to_string(ops.tile.ti) + "," +
                      std::to_string(ops.tile.tj) + "," +
                      std::to_string(ops.tile.tk) + "]";
            size_str = std::to_string(ops.height) + "x" + std::to_string(ops.width);
        } else if (std::holds_alternative<StreamerOperands>(instr.operands)) {
            const auto& ops = std::get<StreamerOperands>(instr.operands);
            matrix_str = (ops.matrix == MatrixID::A) ? "A" :
                        (ops.matrix == MatrixID::B) ? "B" : "C";
            tile_str = "[" + std::to_string(ops.tile.ti) + "," +
                      std::to_string(ops.tile.tj) + "," +
                      std::to_string(ops.tile.tk) + "]";
            size_str = std::to_string(ops.height) + "x" + std::to_string(ops.width);
        }

        std::cout << std::setw(4) << i << " | "
                  << std::setw(20) << std::left << opcode_str << std::right << " | "
                  << std::setw(6) << matrix_str << " | "
                  << std::setw(12) << tile_str << " | "
                  << std::setw(10) << size_str << " | "
                  << instr.label << "\n";
    }
    std::cout << std::string(100, '-') << "\n";
}

// ============================================================================
// Debug: Resource assignment
// ============================================================================

void debug_resource_assignment() {
    print_separator("Debug: Resource Assignment Analysis");

    // Tiny matmul
    OutputStationaryProgramBuilder::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 32;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.L1_Ki = 16;
    config.systolic_size = 16;
    config.element_size = 4;

    config.l3_tile_capacity = 128 * 1024;
    config.l2_bank_capacity = 64 * 1024;
    config.l1_buffer_capacity = 32 * 1024;
    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.num_l1_buffers = 4;
    config.double_buffer = false;

    OutputStationaryProgramBuilder builder(config);
    DMProgram program = builder.build();

    // Configure resources
    ResourceConfig hw_config;
    hw_config.num_memory_channels = 4;
    hw_config.num_block_movers = 4;
    hw_config.num_streamers = 4;
    hw_config.dma_bandwidth_gb_s = 50.0;
    hw_config.block_mover_bandwidth_gb_s = 100.0;
    hw_config.streamer_bandwidth_gb_s = 200.0;

    ConcurrentExecutor executor(hw_config);
    Cycle total_cycles = executor.execute(program);

    std::cout << "\nProgram: " << program.name << "\n";
    std::cout << "Total cycles: " << total_cycles << "\n\n";

    // Analyze scheduled operations
    const auto& ops = executor.get_all_operations();

    std::cout << "Scheduled Operations (sorted by start cycle):\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::setw(6) << "Start" << " | "
              << std::setw(6) << "End" << " | "
              << std::setw(6) << "Dur" << " | "
              << std::setw(10) << "Resource" << " | "
              << std::setw(6) << "Matrix" << " | "
              << std::setw(12) << "Tile" << " | "
              << "Label\n";
    std::cout << std::string(100, '-') << "\n";

    for (const auto& op : ops) {
        std::string matrix_str = (op.matrix == MatrixID::A) ? "A" :
                                (op.matrix == MatrixID::B) ? "B" : "C";
        std::string tile_str = "[" + std::to_string(op.tile.ti) + "," +
                              std::to_string(op.tile.tj) + "," +
                              std::to_string(op.tile.tk) + "]";

        std::cout << std::setw(6) << op.start_cycle << " | "
                  << std::setw(6) << op.end_cycle << " | "
                  << std::setw(6) << op.duration() << " | "
                  << std::setw(10) << op.resource.to_string() << " | "
                  << std::setw(6) << matrix_str << " | "
                  << std::setw(12) << tile_str << " | "
                  << op.label << "\n";
    }
    std::cout << std::string(100, '-') << "\n";

    // Count operations per resource
    std::cout << "\nOperations per resource:\n";
    std::map<std::string, int> resource_counts;
    for (const auto& op : ops) {
        resource_counts[op.resource.to_string()]++;
    }
    for (const auto& [res, count] : resource_counts) {
        std::cout << "  " << res << ": " << count << " operations\n";
    }
}

// ============================================================================
// Debug: Barrier handling
// ============================================================================

void debug_barrier_handling() {
    print_separator("Debug: Barrier and Synchronization Analysis");

    // Tiny matmul
    OutputStationaryProgramBuilder::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 32;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.L1_Ki = 16;
    config.systolic_size = 16;
    config.element_size = 4;

    config.l3_tile_capacity = 128 * 1024;
    config.l2_bank_capacity = 64 * 1024;
    config.l1_buffer_capacity = 32 * 1024;
    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.num_l1_buffers = 4;
    config.double_buffer = false;

    OutputStationaryProgramBuilder builder(config);
    DMProgram program = builder.build();

    std::cout << "\nAnalyzing barrier placement in program:\n\n";

    // Group instructions by barrier sections
    int section = 0;
    std::cout << "Section " << section << ":\n";

    for (size_t i = 0; i < program.instructions.size(); ++i) {
        const auto& instr = program.instructions[i];

        if (instr.opcode == DMOpcode::BARRIER) {
            std::cout << "  --- BARRIER ---\n";
            section++;
            std::cout << "Section " << section << ":\n";
        } else if (instr.opcode == DMOpcode::HALT) {
            std::cout << "  HALT\n";
        } else {
            std::cout << "  [" << i << "] " << instr.label << "\n";
        }
    }

    std::cout << "\nExpected pattern for output-stationary matmul:\n";
    std::cout << "  Section 0: DMA_LOAD A[ti,tk], DMA_LOAD B[tk,tj]\n";
    std::cout << "  BARRIER\n";
    std::cout << "  Section 1: BM_MOVE A, BM_MOVE B\n";
    std::cout << "  BARRIER\n";
    std::cout << "  Section 2: STR_ROWS A, STR_COLS B\n";
    std::cout << "  BARRIER\n";
    std::cout << "  (repeat for each tk in K reduction loop)\n";
    std::cout << "  After K loop: STR_DRAIN C, BARRIER, DMA_STORE C\n";
}

// ============================================================================
// Debug: Expected vs Actual timing
// ============================================================================

void debug_timing_calculation() {
    print_separator("Debug: Timing Calculation Analysis");

    std::cout << "\nBandwidth and cycle calculation:\n\n";

    // Example transfer sizes
    Size tile_size_bytes = 16 * 16 * 4;  // 16x16 tile of float32 = 1024 bytes

    double dma_bw = 50.0;  // GB/s
    double bm_bw = 100.0;
    double str_bw = 200.0;

    // Calculate cycles (assuming 1 GHz clock, so 1 cycle = 1 ns)
    // bytes / (GB/s) = bytes / (10^9 bytes/s) = bytes * 10^-9 s = bytes ns
    // At 1 GHz: cycles = bytes / bandwidth_gb_s

    Cycle dma_cycles = static_cast<Cycle>(tile_size_bytes / dma_bw);
    Cycle bm_cycles = static_cast<Cycle>(tile_size_bytes / bm_bw);
    Cycle str_cycles = static_cast<Cycle>(tile_size_bytes / str_bw);

    std::cout << "Tile size: " << tile_size_bytes << " bytes (16x16 float32)\n\n";

    std::cout << "DMA Engine:\n";
    std::cout << "  Bandwidth: " << dma_bw << " GB/s\n";
    std::cout << "  Cycles for 1 tile: " << dma_cycles << "\n";
    std::cout << "  Issue: " << (dma_cycles == 0 ? "ZERO CYCLES - minimum should be 1!" : "OK") << "\n\n";

    std::cout << "Block Mover:\n";
    std::cout << "  Bandwidth: " << bm_bw << " GB/s\n";
    std::cout << "  Cycles for 1 tile: " << bm_cycles << "\n";
    std::cout << "  Issue: " << (bm_cycles == 0 ? "ZERO CYCLES - minimum should be 1!" : "OK") << "\n\n";

    std::cout << "Streamer:\n";
    std::cout << "  Bandwidth: " << str_bw << " GB/s\n";
    std::cout << "  Cycles for 1 tile: " << str_cycles << "\n";
    std::cout << "  Issue: " << (str_cycles == 0 ? "ZERO CYCLES - minimum should be 1!" : "OK") << "\n\n";

    std::cout << "Problem: With 50 GB/s bandwidth and 1024 bytes:\n";
    std::cout << "  1024 / 50 = 20.48 cycles (truncated to 20)\n";
    std::cout << "  This seems reasonable, but let's verify actual tile sizes...\n\n";

    // Check actual tile sizes in program
    OutputStationaryProgramBuilder::Config config;
    config.M = 32;
    config.N = 32;
    config.K = 32;
    config.Ti = 16;
    config.Tj = 16;
    config.Tk = 16;
    config.L1_Ki = 16;
    config.systolic_size = 16;
    config.element_size = 4;

    config.l3_tile_capacity = 128 * 1024;
    config.l2_bank_capacity = 64 * 1024;
    config.l1_buffer_capacity = 32 * 1024;
    config.num_l3_tiles = 4;
    config.num_l2_banks = 8;
    config.num_l1_buffers = 4;
    config.double_buffer = false;

    OutputStationaryProgramBuilder builder(config);
    DMProgram program = builder.build();

    std::cout << "Actual transfer sizes in program:\n";
    for (size_t i = 0; i < std::min(size_t(10), program.instructions.size()); ++i) {
        const auto& instr = program.instructions[i];
        if (std::holds_alternative<DMAOperands>(instr.operands)) {
            const auto& ops = std::get<DMAOperands>(instr.operands);
            std::cout << "  DMA: " << ops.size_bytes << " bytes\n";
        } else if (std::holds_alternative<BlockMoverOperands>(instr.operands)) {
            const auto& ops = std::get<BlockMoverOperands>(instr.operands);
            Size bytes = ops.height * ops.width * ops.element_size;
            std::cout << "  BM: " << ops.height << "x" << ops.width << "x" << ops.element_size
                      << " = " << bytes << " bytes\n";
        } else if (std::holds_alternative<StreamerOperands>(instr.operands)) {
            const auto& ops = std::get<StreamerOperands>(instr.operands);
            Size bytes = ops.height * ops.width * 4;  // Assumed 4 bytes
            std::cout << "  STR: " << ops.height << "x" << ops.width << " = " << bytes << " bytes\n";
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << R"(
================================================================================
           Concurrent Execution Debug Tool
================================================================================

This tool analyzes the concurrent execution model to find scheduling issues.

Known problems:
1. BM[2], BM[3], STR[2], STR[3] show 0% utilization
2. DMA timeline shows mostly B, rarely A
3. First 30 cycles show only DMA0 with A, nothing else active
4. Barriers may not be sequencing operations correctly

================================================================================
)";

    debug_instruction_scheduling();
    debug_resource_assignment();
    debug_barrier_handling();
    debug_timing_calculation();

    print_separator("Summary of Issues Found");

    std::cout << R"(
Root Cause Analysis:

1. RESOURCE SELECTION HASH:
   The select_dma_channel() function uses:
     hash = matrix * 1000 + ti * 100 + tj * 10 + tk
   For a 32x32x32 with Ti=Tj=Tk=16, we have tiles [0,0,0], [0,0,1], [0,1,0], etc.
   - A[0,0]: hash = 0*1000 + 0*100 + 0*10 + 0 = 0 -> DMA[0]
   - B[0,0]: hash = 1*1000 + 0*100 + 0*10 + 0 = 1000 -> DMA[0] (1000 % 4 = 0)
   This causes DMA[0] to be overloaded!

2. BLOCK MOVER SELECTION:
   Uses src_l3_tile_id % num_block_movers
   But src_l3_tile_id is always 0 in current program generation!

3. STREAMER SELECTION:
   Uses l2_bank_id % num_streamers
   But l2_bank_id is always 0 in current program generation!

4. BARRIER HANDLING:
   Barriers wait for all resources, but operations between barriers
   should run in parallel. Current model may be too sequential.

Fixes needed:
1. Better resource hashing to distribute load
2. Program builder should assign varied L3/L2 IDs
3. Operations between barriers should overlap

)";

    return 0;
}
