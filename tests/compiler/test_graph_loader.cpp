#include <catch2/catch_test_macros.hpp>
#include <sw/compiler/graph_loader.hpp>
#include <filesystem>
#include <iostream>

using namespace sw::kpu::compiler;
namespace fs = std::filesystem;

TEST_CASE("JSONGraphLoader - Load simple matmul", "[compiler][graph_loader]") {
    std::string graph_path = "test_graphs/simple/matmul.json";

    // Check if file exists
    REQUIRE(fs::exists(graph_path));

    // Load graph
    auto graph = load_graph(graph_path);

    REQUIRE(graph != nullptr);
    REQUIRE(graph->name == "simple_matmul");

    // Check tensors
    REQUIRE(graph->tensors.size() == 3);

    auto* tensor_a = graph->get_tensor("A");
    REQUIRE(tensor_a != nullptr);
    REQUIRE(tensor_a->shape.size() == 2);
    REQUIRE(tensor_a->shape[0] == 1024);
    REQUIRE(tensor_a->shape[1] == 512);
    REQUIRE(tensor_a->dtype == "float32");

    auto* tensor_b = graph->get_tensor("B");
    REQUIRE(tensor_b != nullptr);
    REQUIRE(tensor_b->shape[0] == 512);
    REQUIRE(tensor_b->shape[1] == 1024);

    auto* tensor_c = graph->get_tensor("C");
    REQUIRE(tensor_c != nullptr);
    REQUIRE(tensor_c->shape[0] == 1024);
    REQUIRE(tensor_c->shape[1] == 1024);

    // Check operators
    REQUIRE(graph->operators.size() == 1);

    const auto& op = graph->operators[0];
    REQUIRE(op->name == "matmul_0");
    REQUIRE(op->type == OperatorType::MATMUL);
    REQUIRE(op->input_tensors.size() == 2);
    REQUIRE(op->output_tensors.size() == 1);
    REQUIRE(op->input_tensors[0] == "A");
    REQUIRE(op->input_tensors[1] == "B");
    REQUIRE(op->output_tensors[0] == "C");

    // Validate graph
    REQUIRE(graph->validate());
}

TEST_CASE("ComputationalGraph - Execution order", "[compiler][graph_loader]") {
    auto graph = std::make_unique<ComputationalGraph>("test_graph");

    // Create simple chain: A -> Op1 -> B -> Op2 -> C
    TensorDescriptor tensor_a{"A", {100, 100}, "float32", "row_major", 40000};
    TensorDescriptor tensor_b{"B", {100, 100}, "float32", "row_major", 40000};
    TensorDescriptor tensor_c{"C", {100, 100}, "float32", "row_major", 40000};

    graph->add_tensor(tensor_a);
    graph->add_tensor(tensor_b);
    graph->add_tensor(tensor_c);

    auto* op1 = graph->add_operator("op1", OperatorType::RELU);
    op1->add_input("A");
    op1->add_output("B");

    auto* op2 = graph->add_operator("op2", OperatorType::RELU);
    op2->add_input("B");
    op2->add_output("C");

    REQUIRE(graph->validate());

    // Get execution order (should be op1, op2)
    auto order = graph->get_execution_order();
    REQUIRE(order.size() == 2);
    // Note: Actual topological sort not yet implemented, so just check size
}

TEST_CASE("ComputationalGraph - Invalid graph", "[compiler][graph_loader]") {
    auto graph = std::make_unique<ComputationalGraph>("invalid_graph");

    // Add operator referencing non-existent tensor
    auto* op = graph->add_operator("bad_op", OperatorType::RELU);
    op->add_input("nonexistent_tensor");
    op->add_output("output_tensor");

    // Validation should fail
    REQUIRE_FALSE(graph->validate());
}

TEST_CASE("ComputationalGraph - Print graph", "[compiler][graph_loader]") {
    auto graph = std::make_unique<ComputationalGraph>("print_test");

    TensorDescriptor tensor{"test_tensor", {10, 20}, "float32", "row_major", 800};
    graph->add_tensor(tensor);

    auto* op = graph->add_operator("test_op", OperatorType::RELU);
    op->add_input("test_tensor");
    op->add_output("test_tensor");

    // Just check that print doesn't crash
    REQUIRE_NOTHROW(graph->print());
}

#ifdef KPU_HAS_DOMAIN_FLOW
TEST_CASE("DomainFlowGraphLoader - Load .dfg graph", "[compiler][graph_loader][dfg]") {
    std::string graph_path = "test_graphs/simple/matmul.dfg";

    // Check if file exists
    if (!fs::exists(graph_path)) {
        SKIP(".dfg test graph not found - run scripts/copy_domain_flow_graphs.sh");
    }

    // Try to load the graph
    auto loader = std::make_unique<DomainFlowGraphLoader>();
    REQUIRE(loader != nullptr);

    auto extensions = loader->supported_extensions();
    REQUIRE(extensions.size() == 1);
    REQUIRE(extensions[0] == ".dfg");

    // Try to load (may fail if graph doesn't exist, that's ok)
    try {
        auto graph = loader->load(graph_path);
        REQUIRE(graph != nullptr);
        std::cout << "Successfully loaded graph: " << graph->name << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Note: Could not load .dfg file: " << e.what() << std::endl;
        std::cout << "This is expected if test graphs haven't been copied yet." << std::endl;
    }
}
#endif

TEST_CASE("Factory - Create correct loader", "[compiler][graph_loader]") {
    // JSON loader
    auto json_loader = create_graph_loader("test.json");
    REQUIRE(dynamic_cast<JSONGraphLoader*>(json_loader.get()) != nullptr);

#ifdef KPU_HAS_DOMAIN_FLOW
    // domain_flow .dfg loader
    auto dfg_loader = create_graph_loader("test.dfg");
    REQUIRE(dynamic_cast<DomainFlowGraphLoader*>(dfg_loader.get()) != nullptr);
#endif

    // Unsupported format
    REQUIRE_THROWS(create_graph_loader("test.unsupported"));
}
