#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace sw::kpu::compiler {

// Forward declarations
class ComputationalGraph;
class Operator;
class Tensor;
class Schedule;

// Tensor metadata
struct TensorDescriptor {
    std::string name;
    std::vector<size_t> shape;
    std::string dtype;  // "float32", "int8", etc.
    std::string layout; // "NCHW", "NHWC", etc.
    size_t size_bytes;
};

// Operator types
enum class OperatorType {
    GEMM,           // General matrix multiply
    CONV2D,         // 2D convolution
    CONV2D_DEPTHWISE,
    MATMUL,         // Matrix multiplication (synonym for GEMM)
    ELEMENTWISE_ADD,
    ELEMENTWISE_MUL,
    RELU,
    GELU,
    SOFTMAX,
    LAYERNORM,
    BATCHNORM,
    MAXPOOL,
    AVGPOOL,
    RESHAPE,
    TRANSPOSE,
    CONCAT,
    SPLIT,
    ATTENTION,      // Multi-head attention
    UNKNOWN
};

// Operator node in computational graph
class Operator {
public:
    std::string name;
    OperatorType type;
    std::vector<std::string> input_tensors;
    std::vector<std::string> output_tensors;
    std::unordered_map<std::string, std::string> attributes;

    Operator(const std::string& name, OperatorType type)
        : name(name), type(type) {}

    void add_input(const std::string& tensor_name) {
        input_tensors.push_back(tensor_name);
    }

    void add_output(const std::string& tensor_name) {
        output_tensors.push_back(tensor_name);
    }

    void set_attribute(const std::string& key, const std::string& value) {
        attributes[key] = value;
    }
};

// Computational graph representation
class ComputationalGraph {
public:
    std::string name;
    std::vector<std::unique_ptr<Operator>> operators;
    std::unordered_map<std::string, TensorDescriptor> tensors;

    ComputationalGraph() = default;
    ComputationalGraph(const std::string& name) : name(name) {}

    // Add operator to graph
    Operator* add_operator(const std::string& name, OperatorType type) {
        operators.push_back(std::make_unique<Operator>(name, type));
        return operators.back().get();
    }

    // Add tensor descriptor
    void add_tensor(const TensorDescriptor& desc) {
        tensors[desc.name] = desc;
    }

    // Get tensor descriptor
    const TensorDescriptor* get_tensor(const std::string& name) const {
        auto it = tensors.find(name);
        return (it != tensors.end()) ? &it->second : nullptr;
    }

    // Validate graph (check dependencies, types, etc.)
    bool validate() const;

    // Get execution order (topological sort)
    std::vector<Operator*> get_execution_order() const;

    // Print graph for debugging
    void print() const;
};

// Graph loader interface
class GraphLoader {
public:
    virtual ~GraphLoader() = default;

    // Load graph from file
    virtual std::unique_ptr<ComputationalGraph> load(const std::string& filepath) = 0;

    // Get supported file extensions
    virtual std::vector<std::string> supported_extensions() const = 0;
};

#ifdef KPU_HAS_DOMAIN_FLOW
// domain_flow native serialization graph loader
// Uses domain_flow's lightweight serialization format (no LLVM/MLIR dependency)
// File extension: .dfg
class DomainFlowGraphLoader : public GraphLoader {
public:
    std::unique_ptr<ComputationalGraph> load(const std::string& filepath) override;
    std::vector<std::string> supported_extensions() const override {
        return {".dfg"};  // DomainFlowGraph native format
    }

private:
    // Parse domain_flow native format (.dfg)
    bool parse_domain_flow_format(const std::string& filepath, ComputationalGraph* graph);

    // Convert domain_flow operators to KPU operators
    OperatorType convert_operator_type(const std::string& df_op_name);
};
#endif // KPU_HAS_DOMAIN_FLOW

// Simple JSON graph loader (for testing without domain_flow)
class JSONGraphLoader : public GraphLoader {
public:
    std::unique_ptr<ComputationalGraph> load(const std::string& filepath) override;
    std::vector<std::string> supported_extensions() const override {
        return {".json"};
    }
};

// Factory function to get appropriate loader
std::unique_ptr<GraphLoader> create_graph_loader(const std::string& filepath);

// Convenience function to load graph
std::unique_ptr<ComputationalGraph> load_graph(const std::string& filepath);

} // namespace sw::kpu::compiler
