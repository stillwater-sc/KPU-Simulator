/**
 * @file dfg_parser.hpp
 * @brief Domain Flow Graph parser for the KPU kernel compiler
 *
 * Extends the existing GraphLoader to extract additional information
 * needed for KIR generation.
 */

#pragma once

#include <sw/compiler/graph_loader.hpp>
#include <sw/compiler/kir/kir.hpp>
#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace sw::kpu::compiler {

/**
 * @brief Extended information about a matrix operation extracted from DFG
 */
struct MatrixOpInfo {
    // Operation type
    OperatorType op_type;

    // Matrix dimensions for MATMUL: C[M,N] = A[M,K] × B[K,N]
    size_t M;   ///< Rows of A and C
    size_t N;   ///< Columns of B and C
    size_t K;   ///< Columns of A, rows of B

    // Tensor names
    std::string tensor_a;   ///< Name of input A
    std::string tensor_b;   ///< Name of input B
    std::string tensor_c;   ///< Name of output C

    // Optional: Bias tensor for fused operations
    std::optional<std::string> tensor_bias;

    // Data type
    kir::DataType dtype;

    MatrixOpInfo()
        : op_type(OperatorType::MATMUL),
          M(0), N(0), K(0),
          dtype(kir::DataType::FLOAT32) {}
};

/**
 * @brief Parser for Domain Flow Graph files
 *
 * Parses .dfg files and extracts operator information suitable for
 * KIR generation.
 */
class DFGParser {
public:
    /**
     * @brief Parse a DFG file and extract computational graph
     *
     * @param filename Path to .dfg file
     * @return Parsed computational graph (unique_ptr ownership)
     */
    std::unique_ptr<ComputationalGraph> parse(const std::string& filename);

    /**
     * @brief Extract matrix operation info from a computational graph
     *
     * Currently supports:
     * - MATMUL operations
     * - Future: CONV2D, GEMM with bias, etc.
     *
     * @param graph Parsed computational graph
     * @return Vector of matrix operation information
     */
    std::vector<MatrixOpInfo> extract_matrix_ops(const ComputationalGraph& graph);

    /**
     * @brief Get the graph name from the last parsed file
     */
    const std::string& graph_name() const { return graph_name_; }

    /**
     * @brief Get any parsing warnings
     */
    const std::vector<std::string>& warnings() const { return warnings_; }

private:
    std::string graph_name_;
    std::vector<std::string> warnings_;

    /**
     * @brief Parse tensor shape from DFG format
     *
     * Format: "tensor<4x16xf32>" -> shape=[4,16], dtype=f32
     */
    static std::pair<std::vector<size_t>, kir::DataType>
    parse_tensor_type(const std::string& type_str);

    /**
     * @brief Convert OperatorType to KIR DataType based on tensor info
     */
    static kir::DataType infer_dtype(const TensorDescriptor& tensor);
};

} // namespace sw::kpu::compiler
