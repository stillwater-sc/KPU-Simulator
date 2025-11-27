/**
 * @file kir_generator.hpp
 * @brief KIR (KPU Intermediate Representation) generator
 *
 * Transforms parsed DFG operators into hardware-agnostic KIR programs.
 * Uses TileOptimizer to determine tiling parameters and generates
 * the sequence of data movement and compute operations.
 */

#pragma once

#include "dfg_parser.hpp"
#include <sw/compiler/kir/kir.hpp>
#include <sw/compiler/tile_optimizer.hpp>
#include <sw/compiler/l2_tile_scheduler.hpp>

namespace sw::kpu::compiler {

/**
 * @brief Options for KIR generation
 */
struct KIRGeneratorOptions {
    // Dataflow strategy selection
    kir::DataflowStrategy dataflow = kir::DataflowStrategy::OUTPUT_STATIONARY;

    // Tile optimization strategy
    TileOptimizer::Strategy tile_strategy = TileOptimizer::Strategy::ANALYTICAL;

    // Memory hierarchy configuration (affects tile sizing)
    TileOptimizer::MemoryHierarchy memory_hierarchy;

    // Generate prefetch operations
    bool enable_prefetch = true;

    // Verbose output for debugging
    bool verbose = false;

    KIRGeneratorOptions() {
        // Default memory hierarchy from TileOptimizer
    }
};

/**
 * @brief Generates KIR programs from DFG operators
 *
 * This class is the core of the compiler - it transforms high-level
 * operators into sequences of data movement and compute operations
 * that can be executed by the KPU.
 */
class KIRGenerator {
public:
    /**
     * @brief Constructor
     * @param options Generation options
     */
    explicit KIRGenerator(const KIRGeneratorOptions& options = KIRGeneratorOptions());

    /**
     * @brief Generate KIR program from a MATMUL operation
     *
     * @param op_info Matrix operation information
     * @param graph_name Name of the source graph
     * @return Generated KIR program
     */
    kir::Program generate_matmul(const MatrixOpInfo& op_info,
                                  const std::string& graph_name);

    /**
     * @brief Generate KIR program from a complete computational graph
     *
     * Handles multi-operator graphs by generating operations in
     * topological order and managing intermediate tensors.
     *
     * @param graph Parsed computational graph
     * @param ops Matrix operations extracted from graph
     * @return Generated KIR program
     */
    kir::Program generate_program(const ComputationalGraph& graph,
                                   const std::vector<MatrixOpInfo>& ops);

    /**
     * @brief Get generation statistics
     */
    struct GenerationStats {
        size_t num_data_moves;      ///< Number of data movement operations
        size_t num_computes;        ///< Number of compute operations
        size_t num_barriers;        ///< Number of synchronization barriers
        size_t estimated_dram_bytes;
        double estimated_compute_cycles;
    };

    const GenerationStats& stats() const { return stats_; }

    /**
     * @brief Set generation options
     */
    void set_options(const KIRGeneratorOptions& options) { options_ = options; }

    /**
     * @brief Get current options
     */
    const KIRGeneratorOptions& options() const { return options_; }

private:
    KIRGeneratorOptions options_;
    TileOptimizer tile_optimizer_;
    GenerationStats stats_;

    // Operation ID counter
    uint64_t next_op_id_ = 1;

    /**
     * @brief Generate data movement operations for loading tiles
     *
     * Creates the sequence: EXTERNAL → L3 → L2 → L1
     *
     * @param program Program to add operations to
     * @param tensor_name Name of tensor to load
     * @param tile_idx Tile indices [row, col]
     * @param tile_shape Tile shape [height, width]
     * @param dtype Data type
     * @param depends_on Dependencies for the first operation
     * @return Operation ID of the final L1 load
     */
    uint64_t generate_tile_load(kir::Program& program,
                                const std::string& tensor_name,
                                const std::vector<size_t>& tile_idx,
                                const std::vector<size_t>& tile_shape,
                                kir::DataType dtype,
                                const std::vector<uint64_t>& depends_on);

    /**
     * @brief Generate data movement operations for storing tiles
     *
     * Creates the sequence: REGISTER → L1 → L2 → L3 → EXTERNAL
     *
     * @param program Program to add operations to
     * @param tensor_name Name of tensor to store
     * @param tile_idx Tile indices
     * @param tile_shape Tile shape
     * @param dtype Data type
     * @param depends_on Dependencies (typically the compute that produced the tile)
     * @return Operation ID of the final store
     */
    uint64_t generate_tile_store(kir::Program& program,
                                 const std::string& tensor_name,
                                 const std::vector<size_t>& tile_idx,
                                 const std::vector<size_t>& tile_shape,
                                 kir::DataType dtype,
                                 const std::vector<uint64_t>& depends_on);

    /**
     * @brief Generate compute operation for matrix multiplication
     *
     * @param program Program to add operation to
     * @param a_tile A tile specification
     * @param b_tile B tile specification
     * @param c_tile C tile specification
     * @param accumulate Whether to accumulate to existing C
     * @param depends_on Dependencies
     * @return Compute operation ID
     */
    uint64_t generate_matmul_compute(kir::Program& program,
                                     const kir::TileSpec& a_tile,
                                     const kir::TileSpec& b_tile,
                                     const kir::TileSpec& c_tile,
                                     bool accumulate,
                                     const std::vector<uint64_t>& depends_on);

    /**
     * @brief Generate tile iteration loops
     */
    void generate_tile_loops(kir::Program& program,
                             const TileOptimizer::TileConfig& tile_config,
                             size_t M, size_t N, size_t K);

    /**
     * @brief Generate output-stationary schedule
     *
     * Loop order: for ti: for tj: for tk: compute(ti, tj, tk)
     * - Outer loops iterate over C tiles (output stationary)
     * - Inner loop streams through K dimension
     */
    void generate_output_stationary_schedule(kir::Program& program,
                                              const MatrixOpInfo& op_info,
                                              const TileOptimizer::TileConfig& config);

    /**
     * @brief Generate weight-stationary schedule
     *
     * Loop order: for tk: for tj: for ti: compute(ti, tj, tk)
     * - B tiles stay resident longer
     */
    void generate_weight_stationary_schedule(kir::Program& program,
                                              const MatrixOpInfo& op_info,
                                              const TileOptimizer::TileConfig& config);

    /**
     * @brief Add tensor descriptor to program
     */
    void add_tensor(kir::Program& program,
                    const std::string& name,
                    const std::vector<size_t>& shape,
                    kir::DataType dtype,
                    bool is_constant,
                    bool is_output);

    /**
     * @brief Calculate ceiling division
     */
    static size_t ceil_div(size_t a, size_t b) {
        return (a + b - 1) / b;
    }

    /**
     * @brief Get next operation ID
     */
    uint64_t get_next_op_id() { return next_op_id_++; }
};

} // namespace sw::kpu::compiler
