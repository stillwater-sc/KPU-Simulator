#include <sw/compiler/graph_loader.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>

// For JSON parsing
#ifdef KPU_HAS_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

// For domain_flow parsing
#ifdef KPU_HAS_DOMAIN_FLOW
#include <dfa/dfa.hpp>
// Note: We may need util/data_file.hpp for some utilities, but keeping it minimal for now
#endif

namespace sw::kpu::compiler {

// ============================================================================
// ComputationalGraph Implementation
// ============================================================================

bool ComputationalGraph::validate() const {
    // Check all operator inputs reference valid tensors
    for (const auto& op : operators) {
        for (const auto& input : op->input_tensors) {
            if (tensors.find(input) == tensors.end()) {
                std::cerr << "Error: Operator '" << op->name
                         << "' references undefined input tensor '" << input << "'\n";
                return false;
            }
        }
        for (const auto& output : op->output_tensors) {
            if (tensors.find(output) == tensors.end()) {
                std::cerr << "Error: Operator '" << op->name
                         << "' references undefined output tensor '" << output << "'\n";
                return false;
            }
        }
    }
    return true;
}

std::vector<Operator*> ComputationalGraph::get_execution_order() const {
    // Simple topological sort
    // TODO: Implement proper topological sort based on data dependencies
    std::vector<Operator*> order;
    for (const auto& op : operators) {
        order.push_back(op.get());
    }
    return order;
}

void ComputationalGraph::print() const {
    std::cout << "Graph: " << name << "\n";
    std::cout << "Tensors (" << tensors.size() << "):\n";
    for (const auto& [name, desc] : tensors) {
        std::cout << "  " << name << ": [";
        for (size_t i = 0; i < desc.shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << desc.shape[i];
        }
        std::cout << "] " << desc.dtype << "\n";
    }

    std::cout << "Operators (" << operators.size() << "):\n";
    for (const auto& op : operators) {
        std::cout << "  " << op->name << " (type=" << static_cast<int>(op->type) << ")\n";
        std::cout << "    inputs: ";
        for (const auto& in : op->input_tensors) std::cout << in << " ";
        std::cout << "\n    outputs: ";
        for (const auto& out : op->output_tensors) std::cout << out << " ";
        std::cout << "\n";
    }
}

// ============================================================================
// JSONGraphLoader Implementation
// ============================================================================

std::unique_ptr<ComputationalGraph> JSONGraphLoader::load(const std::string& filepath) {
#ifdef KPU_HAS_JSON
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    json j;
    file >> j;

    auto graph = std::make_unique<ComputationalGraph>(j["name"]);

    // Parse tensors
    for (const auto& [name, tensor_json] : j["tensors"].items()) {
        TensorDescriptor desc;
        desc.name = name;
        desc.shape = tensor_json["shape"].get<std::vector<size_t>>();
        desc.dtype = tensor_json["dtype"].get<std::string>();
        desc.layout = tensor_json.value("layout", "row_major");

        // Calculate size in bytes
        size_t element_size = 4; // Default to float32
        if (desc.dtype == "float32" || desc.dtype == "int32") element_size = 4;
        else if (desc.dtype == "float16") element_size = 2;
        else if (desc.dtype == "int8") element_size = 1;

        desc.size_bytes = element_size;
        for (size_t dim : desc.shape) {
            desc.size_bytes *= dim;
        }

        graph->add_tensor(desc);
    }

    // Parse operators
    for (const auto& op_json : j["operators"]) {
        std::string type_str = op_json["type"];
        OperatorType type = OperatorType::UNKNOWN;

        // Convert string to OperatorType
        if (type_str == "MATMUL" || type_str == "GEMM") type = OperatorType::MATMUL;
        else if (type_str == "CONV2D") type = OperatorType::CONV2D;
        else if (type_str == "RELU") type = OperatorType::RELU;
        else if (type_str == "GELU") type = OperatorType::GELU;
        else if (type_str == "ELEMENTWISE_ADD") type = OperatorType::ELEMENTWISE_ADD;
        // Add more operator types as needed

        auto* op = graph->add_operator(op_json["name"], type);

        // Add inputs
        for (const auto& input : op_json["inputs"]) {
            op->add_input(input);
        }

        // Add outputs
        for (const auto& output : op_json["outputs"]) {
            op->add_output(output);
        }

        // Add attributes
        if (op_json.contains("attributes")) {
            for (const auto& [key, value] : op_json["attributes"].items()) {
                op->set_attribute(key, value.dump());
            }
        }
    }

    if (!graph->validate()) {
        throw std::runtime_error("Graph validation failed");
    }

    return graph;
#else
    throw std::runtime_error("JSON support not available (nlohmann_json not found)");
#endif
}

// ============================================================================
// DomainFlowGraphLoader Implementation
// ============================================================================

#ifdef KPU_HAS_DOMAIN_FLOW

std::unique_ptr<ComputationalGraph> DomainFlowGraphLoader::load(const std::string& filepath) {
    // Use domain_flow's native serialization format (no LLVM/MLIR)
    auto graph = std::make_unique<ComputationalGraph>("domain_flow_graph");

    std::cout << "DomainFlowGraphLoader::load() - Using native serialization format\n";
    std::cout << "  File: " << filepath << "\n";

    // Parse domain_flow native format
    if (!parse_domain_flow_format(filepath, graph.get())) {
        throw std::runtime_error("Failed to parse domain_flow format: " + filepath);
    }

    return graph;
}

bool DomainFlowGraphLoader::parse_domain_flow_format(const std::string& filepath,
                                                      ComputationalGraph* graph) {
    try {
        // Create and load domain_flow graph
        sw::dfa::DomainFlowGraph dfg("loaded_graph");
        dfg.load(filepath);

        // Set graph name
        graph->name = dfg.name();

        // Convert tensors
        // TODO: domain_flow API for iterating tensors
        // For now, we'll discover tensors through operators
        std::unordered_set<std::string> tensor_names;

        // Convert operators
        for (size_t i = 0; i < dfg.nrOfNodes(); ++i) {
            const auto& df_node = dfg.node(i);

            // Create KPU operator
            std::string op_name = df_node.name();
            std::string op_type_str = df_node.type();
            OperatorType op_type = convert_operator_type(op_type_str);

            auto* kpu_op = graph->add_operator(op_name, op_type);

            // Add inputs
            for (size_t j = 0; j < df_node.nrOfInputs(); ++j) {
                std::string input_name = df_node.input(j).name();
                kpu_op->add_input(input_name);
                tensor_names.insert(input_name);
            }

            // Add outputs
            for (size_t k = 0; k < df_node.nrOfOutputs(); ++k) {
                std::string output_name = df_node.output(k).name();
                kpu_op->add_output(output_name);
                tensor_names.insert(output_name);
            }

            // Copy attributes
            // TODO: domain_flow attribute iteration API
            // For now, we'll add common attributes we know about
            if (df_node.hasAttribute("transpose_a")) {
                kpu_op->set_attribute("transpose_a", df_node.attribute("transpose_a"));
            }
            if (df_node.hasAttribute("transpose_b")) {
                kpu_op->set_attribute("transpose_b", df_node.attribute("transpose_b"));
            }
        }

        // Create tensor descriptors for discovered tensors
        for (const auto& tensor_name : tensor_names) {
            // TODO: Get actual tensor metadata from domain_flow
            // For now, create placeholder descriptors
            TensorDescriptor desc;
            desc.name = tensor_name;
            desc.dtype = "float32";  // Default
            desc.layout = "row_major";  // Default

            // Try to get shape from domain_flow if available
            // TODO: domain_flow tensor shape API
            desc.shape = {1};  // Placeholder
            desc.size_bytes = 4;  // Placeholder

            graph->add_tensor(desc);
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error parsing domain_flow graph: " << e.what() << std::endl;
        return false;
    }
}

OperatorType DomainFlowGraphLoader::convert_operator_type(const std::string& df_op_name) {
    // Map domain_flow operator names to KPU OperatorType
    if (df_op_name == "matmul" || df_op_name == "gemm") return OperatorType::MATMUL;
    if (df_op_name == "conv2d") return OperatorType::CONV2D;
    if (df_op_name == "relu") return OperatorType::RELU;
    if (df_op_name == "gelu") return OperatorType::GELU;
    // Add more mappings as domain_flow format is defined

    return OperatorType::UNKNOWN;
}

#endif // KPU_HAS_DOMAIN_FLOW

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<GraphLoader> create_graph_loader(const std::string& filepath) {
    // Determine loader based on file extension
    std::string ext;
    size_t dot_pos = filepath.find_last_of('.');
    if (dot_pos != std::string::npos) {
        ext = filepath.substr(dot_pos);
    }

#ifdef KPU_HAS_DOMAIN_FLOW
    // domain_flow native serialization format (.dfg)
    if (ext == ".dfg") {
        return std::make_unique<DomainFlowGraphLoader>();
    }
#endif

    // JSON fallback format
    if (ext == ".json") {
        return std::make_unique<JSONGraphLoader>();
    }

    throw std::runtime_error("Unsupported file format: " + ext);
}

std::unique_ptr<ComputationalGraph> load_graph(const std::string& filepath) {
    auto loader = create_graph_loader(filepath);
    return loader->load(filepath);
}

} // namespace sw::kpu::compiler
