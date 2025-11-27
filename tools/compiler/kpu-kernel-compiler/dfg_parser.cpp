/**
 * @file dfg_parser.cpp
 * @brief Implementation of DFG parser for the KPU kernel compiler
 */

#include "dfg_parser.hpp"
#include <regex>
#include <stdexcept>
#include <iostream>

namespace sw::kpu::compiler {

std::unique_ptr<ComputationalGraph> DFGParser::parse(const std::string& filename) {
    warnings_.clear();

    // Determine file type and create appropriate loader
    std::unique_ptr<GraphLoader> loader;

    if (filename.ends_with(".dfg")) {
#ifdef KPU_HAS_DOMAIN_FLOW
        loader = std::make_unique<DomainFlowGraphLoader>();
#else
        throw std::runtime_error("DFG format support requires KPU_HAS_DOMAIN_FLOW to be defined");
#endif
    } else if (filename.ends_with(".json")) {
        loader = std::make_unique<JSONGraphLoader>();
    } else {
        throw std::runtime_error("Unsupported file format: " + filename +
                                ". Expected .dfg or .json");
    }

    // Load the graph
    auto graph = loader->load(filename);

    // Store the graph name
    graph_name_ = graph->name.empty() ? "unnamed" : graph->name;

    // Validate the graph
    if (!graph->validate()) {
        warnings_.push_back("Graph validation failed");
    }

    return graph;
}

std::vector<MatrixOpInfo> DFGParser::extract_matrix_ops(const ComputationalGraph& graph) {
    std::vector<MatrixOpInfo> ops;

    for (const auto& op : graph.operators) {
        if (op->type == OperatorType::MATMUL || op->type == OperatorType::GEMM) {
            MatrixOpInfo info;
            info.op_type = op->type;

            // Extract input tensor names
            if (op->input_tensors.size() >= 2) {
                info.tensor_a = op->input_tensors[0];
                info.tensor_b = op->input_tensors[1];
            } else {
                warnings_.push_back("MATMUL operation '" + op->name +
                                   "' has fewer than 2 inputs");
                continue;
            }

            // Extract output tensor name
            if (!op->output_tensors.empty()) {
                info.tensor_c = op->output_tensors[0];
            } else {
                warnings_.push_back("MATMUL operation '" + op->name +
                                   "' has no outputs");
                continue;
            }

            // Check for bias (GEMM with 3 inputs)
            if (op->input_tensors.size() >= 3) {
                info.tensor_bias = op->input_tensors[2];
            }

            // Get tensor shapes
            auto it_a = graph.tensors.find(info.tensor_a);
            auto it_b = graph.tensors.find(info.tensor_b);
            auto it_c = graph.tensors.find(info.tensor_c);

            if (it_a == graph.tensors.end() ||
                it_b == graph.tensors.end() ||
                it_c == graph.tensors.end()) {
                warnings_.push_back("Could not find tensors for MATMUL operation '" +
                                   op->name + "'");
                continue;
            }

            const auto& tensor_a = it_a->second;
            const auto& tensor_b = it_b->second;
            // const auto& tensor_c = it_c->second;

            // Extract dimensions from shapes
            // A[M,K] × B[K,N] = C[M,N]
            if (tensor_a.shape.size() >= 2 && tensor_b.shape.size() >= 2) {
                info.M = tensor_a.shape[tensor_a.shape.size() - 2];
                info.K = tensor_a.shape[tensor_a.shape.size() - 1];
                info.N = tensor_b.shape[tensor_b.shape.size() - 1];

                // Verify K dimension matches
                size_t b_k = tensor_b.shape[tensor_b.shape.size() - 2];
                if (info.K != b_k) {
                    warnings_.push_back("Dimension mismatch in MATMUL '" + op->name +
                                       "': A.cols=" + std::to_string(info.K) +
                                       " != B.rows=" + std::to_string(b_k));
                }
            } else {
                warnings_.push_back("Unexpected tensor shapes in MATMUL '" + op->name + "'");
                continue;
            }

            // Infer data type
            info.dtype = infer_dtype(tensor_a);

            ops.push_back(info);
        }
        // TODO: Add support for CONV2D and other operations
    }

    return ops;
}

std::pair<std::vector<size_t>, kir::DataType>
DFGParser::parse_tensor_type(const std::string& type_str) {
    std::vector<size_t> shape;
    kir::DataType dtype = kir::DataType::FLOAT32;

    // Pattern: tensor<4x16xf32>
    std::regex tensor_regex(R"(tensor<(.+)>)");
    std::smatch match;

    if (std::regex_match(type_str, match, tensor_regex)) {
        std::string dims_str = match[1].str();

        // Split by 'x' and parse
        std::regex dim_regex(R"((\d+)x?)");
        auto begin = std::sregex_iterator(dims_str.begin(), dims_str.end(), dim_regex);
        auto end = std::sregex_iterator();

        for (auto it = begin; it != end; ++it) {
            std::string dim = (*it)[1].str();
            shape.push_back(std::stoull(dim));
        }

        // Extract dtype suffix
        if (dims_str.find("f32") != std::string::npos) {
            dtype = kir::DataType::FLOAT32;
        } else if (dims_str.find("f16") != std::string::npos) {
            dtype = kir::DataType::FLOAT16;
        } else if (dims_str.find("bf16") != std::string::npos) {
            dtype = kir::DataType::BFLOAT16;
        } else if (dims_str.find("i32") != std::string::npos) {
            dtype = kir::DataType::INT32;
        } else if (dims_str.find("i8") != std::string::npos) {
            dtype = kir::DataType::INT8;
        }
    }

    return {shape, dtype};
}

kir::DataType DFGParser::infer_dtype(const TensorDescriptor& tensor) {
    // Use the tensor's dtype string
    const std::string& dtype = tensor.dtype;

    if (dtype == "float32" || dtype == "f32") {
        return kir::DataType::FLOAT32;
    } else if (dtype == "float16" || dtype == "f16") {
        return kir::DataType::FLOAT16;
    } else if (dtype == "bfloat16" || dtype == "bf16") {
        return kir::DataType::BFLOAT16;
    } else if (dtype == "int32" || dtype == "i32") {
        return kir::DataType::INT32;
    } else if (dtype == "int16" || dtype == "i16") {
        return kir::DataType::INT16;
    } else if (dtype == "int8" || dtype == "i8") {
        return kir::DataType::INT8;
    } else if (dtype == "uint8" || dtype == "u8") {
        return kir::DataType::UINT8;
    }

    // Default to float32
    return kir::DataType::FLOAT32;
}

} // namespace sw::kpu::compiler
