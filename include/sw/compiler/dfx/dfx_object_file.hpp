/**
 * @file dfx_object_file.hpp
 * @brief KPU Object File (.kpu) format definitions
 *
 * Defines the serialization format for DFX programs. The object file
 * contains all information needed to load and execute a kernel on
 * the KPU simulator.
 *
 * File Format:
 * - JSON-based for initial implementation (human-readable, debuggable)
 * - Binary format planned for production use
 */

#pragma once

#include <sw/compiler/dfx/dfx.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <fstream>
#include <stdexcept>

namespace sw::kpu::compiler::dfx {

// Use nlohmann::json for serialization
using json = nlohmann::json;

// ============================================================================
// JSON Serialization Helpers
// ============================================================================

/**
 * @brief Serialize DataType to JSON
 */
inline void to_json(json& j, const DataType& dtype) {
    j = dtype_to_string(dtype);
}

/**
 * @brief Deserialize DataType from JSON
 */
inline void from_json(const json& j, DataType& dtype) {
    std::string s = j.get<std::string>();
    if (s == "f32") dtype = DataType::FLOAT32;
    else if (s == "f16") dtype = DataType::FLOAT16;
    else if (s == "bf16") dtype = DataType::BFLOAT16;
    else if (s == "i32") dtype = DataType::INT32;
    else if (s == "i16") dtype = DataType::INT16;
    else if (s == "i8") dtype = DataType::INT8;
    else if (s == "u8") dtype = DataType::UINT8;
    else if (s == "bool") dtype = DataType::BOOL;
    else throw std::runtime_error("Unknown data type: " + s);
}

/**
 * @brief Serialize MemoryLevel to JSON
 */
inline void to_json(json& j, const MemoryLevel& level) {
    j = memory_level_to_string(level);
}

/**
 * @brief Deserialize MemoryLevel from JSON
 */
inline void from_json(const json& j, MemoryLevel& level) {
    std::string s = j.get<std::string>();
    if (s == "EXTERNAL") level = MemoryLevel::EXTERNAL;
    else if (s == "L3") level = MemoryLevel::L3;
    else if (s == "L2") level = MemoryLevel::L2;
    else if (s == "L1") level = MemoryLevel::L1;
    else if (s == "REGISTER") level = MemoryLevel::REGISTER;
    else throw std::runtime_error("Unknown memory level: " + s);
}

/**
 * @brief Serialize DataflowStrategy to JSON
 */
inline void to_json(json& j, const DataflowStrategy& strategy) {
    j = dataflow_to_string(strategy);
}

/**
 * @brief Deserialize DataflowStrategy from JSON
 */
inline void from_json(const json& j, DataflowStrategy& strategy) {
    std::string s = j.get<std::string>();
    if (s == "output_stationary") strategy = DataflowStrategy::OUTPUT_STATIONARY;
    else if (s == "weight_stationary") strategy = DataflowStrategy::WEIGHT_STATIONARY;
    else if (s == "input_stationary") strategy = DataflowStrategy::INPUT_STATIONARY;
    else throw std::runtime_error("Unknown dataflow strategy: " + s);
}

/**
 * @brief Serialize TensorDescriptor to JSON
 */
inline void to_json(json& j, const TensorDescriptor& tensor) {
    j = json{
        {"name", tensor.name},
        {"shape", tensor.shape},
        {"dtype", tensor.dtype},
        {"is_constant", tensor.is_constant},
        {"is_output", tensor.is_output}
    };
}

/**
 * @brief Deserialize TensorDescriptor from JSON
 */
inline void from_json(const json& j, TensorDescriptor& tensor) {
    tensor.name = j.at("name").get<std::string>();
    tensor.shape = j.at("shape").get<std::vector<size_t>>();
    tensor.dtype = j.at("dtype").get<DataType>();
    tensor.is_constant = j.value("is_constant", false);
    tensor.is_output = j.value("is_output", false);
}

/**
 * @brief Serialize TileSpec to JSON
 */
inline void to_json(json& j, const TileSpec& tile) {
    j = json{
        {"tensor", tile.tensor_name},
        {"level", tile.level},
        {"tile_indices", tile.tile_indices},
        {"tile_shape", tile.tile_shape}
    };
}

/**
 * @brief Deserialize TileSpec from JSON
 */
inline void from_json(const json& j, TileSpec& tile) {
    tile.tensor_name = j.at("tensor").get<std::string>();
    tile.level = j.at("level").get<MemoryLevel>();
    tile.tile_indices = j.at("tile_indices").get<std::vector<size_t>>();
    tile.tile_shape = j.at("tile_shape").get<std::vector<size_t>>();
}

/**
 * @brief Serialize TilingConfig to JSON
 */
inline void to_json(json& j, const TilingConfig& config) {
    j = json{
        {"Ti", config.Ti},
        {"Tj", config.Tj},
        {"Tk", config.Tk},
        {"L1_Ki", config.L1_Ki},
        {"num_tiles_m", config.num_tiles_m},
        {"num_tiles_n", config.num_tiles_n},
        {"num_tiles_k", config.num_tiles_k}
    };
}

/**
 * @brief Deserialize TilingConfig from JSON
 */
inline void from_json(const json& j, TilingConfig& config) {
    config.Ti = j.at("Ti").get<size_t>();
    config.Tj = j.at("Tj").get<size_t>();
    config.Tk = j.at("Tk").get<size_t>();
    config.L1_Ki = j.value("L1_Ki", size_t(0));
    config.num_tiles_m = j.value("num_tiles_m", size_t(0));
    config.num_tiles_n = j.value("num_tiles_n", size_t(0));
    config.num_tiles_k = j.value("num_tiles_k", size_t(0));
}

/**
 * @brief Serialize PerformanceHints to JSON
 */
inline void to_json(json& j, const PerformanceHints& hints) {
    j = json{
        {"estimated_dram_bytes", hints.estimated_dram_bytes},
        {"estimated_compute_cycles", hints.estimated_compute_cycles},
        {"arithmetic_intensity", hints.arithmetic_intensity},
        {"parallelism_degree", hints.parallelism_degree},
        {"reuse_factor_a", hints.reuse_factor_a},
        {"reuse_factor_b", hints.reuse_factor_b}
    };
}

/**
 * @brief Deserialize PerformanceHints from JSON
 */
inline void from_json(const json& j, PerformanceHints& hints) {
    hints.estimated_dram_bytes = j.value("estimated_dram_bytes", size_t(0));
    hints.estimated_compute_cycles = j.value("estimated_compute_cycles", size_t(0));
    hints.arithmetic_intensity = j.value("arithmetic_intensity", 0.0);
    hints.parallelism_degree = j.value("parallelism_degree", size_t(1));
    hints.reuse_factor_a = j.value("reuse_factor_a", size_t(1));
    hints.reuse_factor_b = j.value("reuse_factor_b", size_t(1));
}

/**
 * @brief Serialize TileLoop to JSON
 */
inline void to_json(json& j, const TileLoop& loop) {
    j = json{
        {"induction_var", loop.induction_var},
        {"start", loop.start},
        {"end", loop.end},
        {"step", loop.step}
    };
    if (!loop.iteration_order.empty()) {
        j["iteration_order"] = loop.iteration_order;
    }
}

/**
 * @brief Deserialize TileLoop from JSON
 */
inline void from_json(const json& j, TileLoop& loop) {
    loop.induction_var = j.at("induction_var").get<std::string>();
    loop.start = j.at("start").get<size_t>();
    loop.end = j.at("end").get<size_t>();
    loop.step = j.value("step", size_t(1));
    if (j.contains("iteration_order")) {
        loop.iteration_order = j.at("iteration_order").get<std::vector<size_t>>();
    }
}

// ============================================================================
// Operation Serialization
// ============================================================================

/**
 * @brief Serialize an operation to JSON
 */
inline json operation_to_json(const Operation& op) {
    json j;
    j["op_id"] = op.op_id;
    j["depends_on"] = op.depends_on;
    if (!op.label.empty()) {
        j["label"] = op.label;
    }

    if (auto* data_move = dynamic_cast<const DataMoveOp*>(&op)) {
        j["type"] = "DATA_MOVE";
        j["move_type"] = data_move_type_to_string(data_move->move_type);
        j["source"] = data_move->source;
        j["destination"] = data_move->destination;
        j["transpose"] = data_move->transpose;
        j["broadcast"] = data_move->broadcast;
    }
    else if (auto* compute = dynamic_cast<const ComputeOp*>(&op)) {
        j["type"] = "COMPUTE";
        j["compute_type"] = compute_type_to_string(compute->compute_type);
        j["inputs"] = compute->inputs;
        j["output"] = compute->output;
        j["accumulate"] = compute->accumulate;
        if (compute->reduction_axis.has_value()) {
            j["reduction_axis"] = compute->reduction_axis.value();
        }
    }
    else if (auto* barrier = dynamic_cast<const BarrierOp*>(&op)) {
        j["type"] = "BARRIER";
        j["wait_for"] = barrier->wait_for;
    }

    return j;
}

/**
 * @brief Deserialize an operation from JSON
 */
inline std::unique_ptr<Operation> operation_from_json(const json& j) {
    std::string type = j.at("type").get<std::string>();

    if (type == "DATA_MOVE") {
        auto op = std::make_unique<DataMoveOp>();
        op->op_id = j.at("op_id").get<uint64_t>();
        op->depends_on = j.value("depends_on", std::vector<uint64_t>{});
        op->label = j.value("label", std::string{});

        std::string move_type_str = j.at("move_type").get<std::string>();
        if (move_type_str == "LOAD") op->move_type = DataMoveType::LOAD;
        else if (move_type_str == "STORE") op->move_type = DataMoveType::STORE;
        else if (move_type_str == "PREFETCH") op->move_type = DataMoveType::PREFETCH;
        else if (move_type_str == "FLUSH") op->move_type = DataMoveType::FLUSH;

        op->source = j.at("source").get<TileSpec>();
        op->destination = j.at("destination").get<TileSpec>();
        op->transpose = j.value("transpose", false);
        op->broadcast = j.value("broadcast", false);

        return op;
    }
    else if (type == "COMPUTE") {
        auto op = std::make_unique<ComputeOp>();
        op->op_id = j.at("op_id").get<uint64_t>();
        op->depends_on = j.value("depends_on", std::vector<uint64_t>{});
        op->label = j.value("label", std::string{});

        std::string compute_type_str = j.at("compute_type").get<std::string>();
        if (compute_type_str == "MATMUL_TILE") op->compute_type = ComputeType::MATMUL_TILE;
        else if (compute_type_str == "CONV2D_TILE") op->compute_type = ComputeType::CONV2D_TILE;
        else if (compute_type_str == "ELEMENTWISE_ADD") op->compute_type = ComputeType::ELEMENTWISE_ADD;
        else if (compute_type_str == "ELEMENTWISE_MUL") op->compute_type = ComputeType::ELEMENTWISE_MUL;
        else if (compute_type_str == "ELEMENTWISE_SUB") op->compute_type = ComputeType::ELEMENTWISE_SUB;
        else if (compute_type_str == "RELU") op->compute_type = ComputeType::RELU;
        else if (compute_type_str == "GELU") op->compute_type = ComputeType::GELU;
        else if (compute_type_str == "SIGMOID") op->compute_type = ComputeType::SIGMOID;
        else if (compute_type_str == "TANH") op->compute_type = ComputeType::TANH;
        else if (compute_type_str == "SOFTMAX_TILE") op->compute_type = ComputeType::SOFTMAX_TILE;
        else if (compute_type_str == "LAYERNORM_TILE") op->compute_type = ComputeType::LAYERNORM_TILE;
        else if (compute_type_str == "REDUCE_SUM") op->compute_type = ComputeType::REDUCE_SUM;
        else if (compute_type_str == "REDUCE_MAX") op->compute_type = ComputeType::REDUCE_MAX;
        else if (compute_type_str == "REDUCE_MEAN") op->compute_type = ComputeType::REDUCE_MEAN;

        op->inputs = j.at("inputs").get<std::vector<TileSpec>>();
        op->output = j.at("output").get<TileSpec>();
        op->accumulate = j.value("accumulate", false);
        if (j.contains("reduction_axis")) {
            op->reduction_axis = j.at("reduction_axis").get<int>();
        }

        return op;
    }
    else if (type == "BARRIER") {
        auto op = std::make_unique<BarrierOp>();
        op->op_id = j.at("op_id").get<uint64_t>();
        op->depends_on = j.value("depends_on", std::vector<uint64_t>{});
        op->label = j.value("label", std::string{});
        op->wait_for = j.at("wait_for").get<std::vector<uint64_t>>();

        return op;
    }

    throw std::runtime_error("Unknown operation type: " + type);
}

// ============================================================================
// Program Serialization
// ============================================================================

/**
 * @brief Serialize a DFX Program to JSON
 */
inline json program_to_json(const Program& program) {
    json j;

    // Metadata
    j["dfx_version"] = dfx_version_string();
    j["name"] = program.name;
    j["source_graph"] = program.source_graph;

    // Configuration
    j["dataflow"] = program.dataflow;
    j["tiling"] = program.tiling;

    // Tensors
    j["tensors"] = program.tensors;

    // Operations
    json ops_array = json::array();
    for (const auto& op : program.operations) {
        ops_array.push_back(operation_to_json(*op));
    }
    j["operations"] = ops_array;

    // Tile loops
    j["tile_loops"] = program.tile_loops;

    // Performance hints
    j["hints"] = program.hints;

    return j;
}

/**
 * @brief Deserialize a DFX Program from JSON
 */
inline Program program_from_json(const json& j) {
    Program program;

    // Metadata
    program.name = j.at("name").get<std::string>();
    program.source_graph = j.value("source_graph", std::string{});

    // Parse version if present (support both old kir_version and new dfx_version)
    if (j.contains("dfx_version") || j.contains("kir_version")) {
        // Could validate version here
    }

    // Configuration
    program.dataflow = j.at("dataflow").get<DataflowStrategy>();
    program.tiling = j.at("tiling").get<TilingConfig>();

    // Tensors
    program.tensors = j.at("tensors").get<std::vector<TensorDescriptor>>();

    // Operations
    for (const auto& op_json : j.at("operations")) {
        program.operations.push_back(operation_from_json(op_json));
    }

    // Tile loops
    if (j.contains("tile_loops")) {
        program.tile_loops = j.at("tile_loops").get<std::vector<TileLoop>>();
    }

    // Performance hints
    if (j.contains("hints")) {
        program.hints = j.at("hints").get<PerformanceHints>();
    }

    return program;
}

// ============================================================================
// Object File I/O
// ============================================================================

/**
 * @brief KPU Object File magic number
 */
constexpr const char* KPU_OBJECT_MAGIC = "KPU";

/**
 * @brief Write a DFX program to a .kpu object file
 *
 * @param program The program to write
 * @param filename Output filename (should end in .kpu)
 * @param pretty Whether to pretty-print JSON (for debugging)
 */
inline void write_object_file(const Program& program,
                              const std::string& filename,
                              bool pretty = true) {
    json j = program_to_json(program);

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    if (pretty) {
        file << j.dump(2);  // 2-space indentation
    } else {
        file << j.dump();
    }
}

/**
 * @brief Read a DFX program from a .kpu object file
 *
 * @param filename Input filename (should end in .kpu)
 * @return Loaded program
 */
inline Program read_object_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    json j;
    file >> j;

    return program_from_json(j);
}

} // namespace sw::kpu::compiler::dfx
