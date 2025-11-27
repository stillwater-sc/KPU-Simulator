/**
 * @file disassembler.cpp
 * @brief KPU Disassembler - Object file disassembler
 *
 * Reads .kpu object files and prints human-readable DFX.
 */

#include <sw/compiler/dfx/dfx.hpp>
#include <sw/compiler/dfx/dfx_object_file.hpp>
#include <iostream>
#include <string>

using namespace sw::kpu::compiler::dfx;

void print_usage(const char* program_name) {
    std::cout << "KPU Disassembler - Disassembles KPU object files\n\n";
    std::cout << "Usage: " << program_name << " PROGRAM.kpu [options]\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string filename = argv[1];

    if (filename == "-h" || filename == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    try {
        auto program = read_object_file(filename);

        std::cout << "=== KPU Object File: " << filename << " ===\n\n";
        std::cout << "Name: " << program.name << "\n";
        std::cout << "Source: " << program.source_graph << "\n";
        std::cout << "DFX Version: " << program.version_major << "."
                  << program.version_minor << "." << program.version_patch << "\n";
        std::cout << "Dataflow: " << dataflow_to_string(program.dataflow) << "\n";

        std::cout << "\nTiling:\n";
        std::cout << "  Ti=" << program.tiling.Ti
                  << ", Tj=" << program.tiling.Tj
                  << ", Tk=" << program.tiling.Tk << "\n";
        std::cout << "  Grid: " << program.tiling.num_tiles_m << " x "
                  << program.tiling.num_tiles_n << " x "
                  << program.tiling.num_tiles_k << " tiles\n";

        std::cout << "\nTensors (" << program.tensors.size() << "):\n";
        for (const auto& t : program.tensors) {
            std::cout << "  " << t.name << ": [";
            for (size_t i = 0; i < t.shape.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << t.shape[i];
            }
            std::cout << "] " << dtype_to_string(t.dtype);
            if (t.is_constant) std::cout << " (const)";
            if (t.is_output) std::cout << " (output)";
            std::cout << "\n";
        }

        std::cout << "\nOperations (" << program.operations.size() << "):\n";
        for (const auto& op : program.operations) {
            std::cout << "  [" << op->op_id << "] " << op->type_name();
            if (!op->label.empty()) {
                std::cout << " - " << op->label;
            }
            if (!op->depends_on.empty()) {
                std::cout << " (deps: ";
                for (size_t i = 0; i < op->depends_on.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << op->depends_on[i];
                }
                std::cout << ")";
            }
            std::cout << "\n";
        }

        std::cout << "\nPerformance Hints:\n";
        std::cout << "  Estimated DRAM bytes: " << program.hints.estimated_dram_bytes << "\n";
        std::cout << "  Estimated compute cycles: " << program.hints.estimated_compute_cycles << "\n";
        std::cout << "  Arithmetic intensity: " << program.hints.arithmetic_intensity << " FLOP/byte\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
