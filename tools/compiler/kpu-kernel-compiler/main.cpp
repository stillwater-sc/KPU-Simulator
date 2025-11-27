/**
 * @file main.cpp
 * @brief KPU Kernel Compiler - Main entry point
 *
 * Compiles Domain Flow Graph (.dfg) files to KPU object files (.kpu).
 *
 * Usage:
 *   kpu-kernel-compiler input.dfg -o output.kpu [options]
 *
 * Options:
 *   -o, --output FILE      Output object file (.kpu)
 *   -d, --dataflow STRATEGY Dataflow strategy (output/weight/input stationary)
 *   -t, --tile-strategy    Tile optimization strategy (analytical/search)
 *   --emit-kir             Print KIR to stdout
 *   --dump                 Dump parsed graph information
 *   -v, --verbose          Verbose output
 *   -h, --help             Show help message
 */

#include "dfg_parser.hpp"
#include "kir_generator.hpp"
#include "object_writer.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

// Simple command-line argument parsing
struct CommandLineArgs {
    std::string input_file;
    std::string output_file;
    std::string dataflow = "output-stationary";
    std::string tile_strategy = "analytical";
    bool emit_kir = false;
    bool dump_graph = false;
    bool verbose = false;
    bool help = false;
    bool error = false;
    std::string error_message;
};

void print_usage(const char* program_name) {
    std::cout << "KPU Kernel Compiler - Compiles Domain Flow Graphs to KPU object files\n\n";
    std::cout << "Usage: " << program_name << " INPUT.dfg [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -o, --output FILE       Output object file (.kpu)\n";
    std::cout << "  -d, --dataflow STRATEGY Dataflow strategy:\n";
    std::cout << "                          output-stationary (default)\n";
    std::cout << "                          weight-stationary\n";
    std::cout << "                          input-stationary\n";
    std::cout << "  -t, --tile-strategy STR Tile optimization strategy:\n";
    std::cout << "                          analytical (default)\n";
    std::cout << "                          search\n";
    std::cout << "                          heuristic\n";
    std::cout << "  --emit-kir              Print KIR to stdout\n";
    std::cout << "  --dump                  Dump parsed graph information\n";
    std::cout << "  -v, --verbose           Verbose output\n";
    std::cout << "  -h, --help              Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " matmul.dfg -o matmul.kpu\n";
    std::cout << "  " << program_name << " matmul.dfg --emit-kir\n";
    std::cout << "  " << program_name << " conv2d.dfg -d weight-stationary -o conv2d.kpu\n";
}

CommandLineArgs parse_args(int argc, char* argv[]) {
    CommandLineArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            args.help = true;
            return args;
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                args.output_file = argv[++i];
            } else {
                args.error = true;
                args.error_message = "Missing argument for " + arg;
                return args;
            }
        }
        else if (arg == "-d" || arg == "--dataflow") {
            if (i + 1 < argc) {
                args.dataflow = argv[++i];
            } else {
                args.error = true;
                args.error_message = "Missing argument for " + arg;
                return args;
            }
        }
        else if (arg == "-t" || arg == "--tile-strategy") {
            if (i + 1 < argc) {
                args.tile_strategy = argv[++i];
            } else {
                args.error = true;
                args.error_message = "Missing argument for " + arg;
                return args;
            }
        }
        else if (arg == "--emit-kir") {
            args.emit_kir = true;
        }
        else if (arg == "--dump") {
            args.dump_graph = true;
        }
        else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        }
        else if (arg[0] == '-') {
            args.error = true;
            args.error_message = "Unknown option: " + arg;
            return args;
        }
        else {
            if (args.input_file.empty()) {
                args.input_file = arg;
            } else {
                args.error = true;
                args.error_message = "Multiple input files specified";
                return args;
            }
        }
    }

    // Validate required arguments
    if (args.input_file.empty() && !args.help) {
        args.error = true;
        args.error_message = "No input file specified";
    }

    // Generate default output filename if not specified
    if (args.output_file.empty() && !args.input_file.empty()) {
        fs::path input_path(args.input_file);
        args.output_file = input_path.stem().string() + ".kpu";
    }

    return args;
}

sw::kpu::compiler::kir::DataflowStrategy parse_dataflow(const std::string& str) {
    if (str == "output-stationary" || str == "os") {
        return sw::kpu::compiler::kir::DataflowStrategy::OUTPUT_STATIONARY;
    } else if (str == "weight-stationary" || str == "ws") {
        return sw::kpu::compiler::kir::DataflowStrategy::WEIGHT_STATIONARY;
    } else if (str == "input-stationary" || str == "is") {
        return sw::kpu::compiler::kir::DataflowStrategy::INPUT_STATIONARY;
    }
    throw std::runtime_error("Unknown dataflow strategy: " + str);
}

sw::kpu::compiler::TileOptimizer::Strategy parse_tile_strategy(const std::string& str) {
    if (str == "analytical") {
        return sw::kpu::compiler::TileOptimizer::Strategy::ANALYTICAL;
    } else if (str == "search") {
        return sw::kpu::compiler::TileOptimizer::Strategy::BOUNDED_SEARCH;
    } else if (str == "heuristic") {
        return sw::kpu::compiler::TileOptimizer::Strategy::HEURISTIC_HYBRID;
    }
    throw std::runtime_error("Unknown tile strategy: " + str);
}

int main(int argc, char* argv[]) {
    CommandLineArgs args = parse_args(argc, argv);

    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }

    if (args.error) {
        std::cerr << "Error: " << args.error_message << "\n";
        std::cerr << "Use --help for usage information.\n";
        return 1;
    }

    try {
        if (args.verbose) {
            std::cout << "KPU Kernel Compiler v" << sw::kpu::compiler::kir::kir_version_string() << "\n";
            std::cout << "Input: " << args.input_file << "\n";
            std::cout << "Output: " << args.output_file << "\n";
            std::cout << "Dataflow: " << args.dataflow << "\n";
            std::cout << "Tile strategy: " << args.tile_strategy << "\n";
            std::cout << "\n";
        }

        // Parse the input file
        if (args.verbose) {
            std::cout << "Parsing " << args.input_file << "...\n";
        }

        sw::kpu::compiler::DFGParser parser;
        auto graph = parser.parse(args.input_file);

        // Print any warnings
        for (const auto& warning : parser.warnings()) {
            std::cerr << "Warning: " << warning << "\n";
        }

        // Dump graph info if requested
        if (args.dump_graph) {
            std::cout << "\n=== Graph Information ===\n";
            std::cout << "Name: " << graph->name << "\n";
            std::cout << "Operators: " << graph->operators.size() << "\n";
            std::cout << "Tensors: " << graph->tensors.size() << "\n";

            std::cout << "\nTensors:\n";
            for (const auto& [name, tensor] : graph->tensors) {
                std::cout << "  " << name << ": [";
                for (size_t i = 0; i < tensor.shape.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << tensor.shape[i];
                }
                std::cout << "]\n";
            }

            std::cout << "\nOperators:\n";
            for (const auto& op : graph->operators) {
                std::cout << "  " << op->name << " (" << static_cast<int>(op->type) << ")\n";
                std::cout << "    Inputs: ";
                for (size_t i = 0; i < op->input_tensors.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << op->input_tensors[i];
                }
                std::cout << "\n    Outputs: ";
                for (size_t i = 0; i < op->output_tensors.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << op->output_tensors[i];
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        // Extract matrix operations
        auto matrix_ops = parser.extract_matrix_ops(*graph);

        if (matrix_ops.empty()) {
            std::cerr << "Error: No matrix operations found in graph\n";
            return 1;
        }

        if (args.verbose) {
            std::cout << "Found " << matrix_ops.size() << " matrix operation(s)\n";
            for (const auto& op : matrix_ops) {
                std::cout << "  MATMUL: " << op.tensor_c << " = "
                          << op.tensor_a << " × " << op.tensor_b << "\n";
                std::cout << "    Dimensions: M=" << op.M << ", N=" << op.N << ", K=" << op.K << "\n";
            }
        }

        // Configure KIR generator
        sw::kpu::compiler::KIRGeneratorOptions gen_options;
        gen_options.dataflow = parse_dataflow(args.dataflow);
        gen_options.tile_strategy = parse_tile_strategy(args.tile_strategy);
        gen_options.verbose = args.verbose;

        // Generate KIR program
        if (args.verbose) {
            std::cout << "\nGenerating KIR...\n";
        }

        sw::kpu::compiler::KIRGenerator generator(gen_options);
        auto program = generator.generate_program(*graph, matrix_ops);

        // Emit KIR to stdout if requested
        if (args.emit_kir) {
            sw::kpu::compiler::ObjectWriter writer;
            std::cout << "\n=== KIR Program ===\n";
            std::cout << writer.to_string(program);
            std::cout << "\n";
        }

        // Write object file
        if (!args.emit_kir || !args.output_file.empty()) {
            if (args.verbose) {
                std::cout << "Writing " << args.output_file << "...\n";
            }

            sw::kpu::compiler::ObjectWriter writer;
            writer.write(program, args.output_file);

            std::cout << "Compiled " << args.input_file << " -> " << args.output_file << "\n";
        }

        // Print summary
        const auto& stats = generator.stats();
        if (args.verbose) {
            std::cout << "\n=== Compilation Summary ===\n";
            std::cout << "Data movement operations: " << stats.num_data_moves << "\n";
            std::cout << "Compute operations: " << stats.num_computes << "\n";
            std::cout << "Estimated DRAM traffic: " << stats.estimated_dram_bytes << " bytes\n";
            std::cout << "Estimated compute cycles: " << stats.estimated_compute_cycles << "\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
