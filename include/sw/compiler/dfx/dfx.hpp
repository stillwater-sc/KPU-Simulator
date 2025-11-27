/**
 * @file dfx.hpp
 * @brief Domain Flow Execution (DFX) - The PTX-equivalent layer for KPU
 *
 * DFX is the hardware-agnostic intermediate representation for KPU programs.
 * Like NVIDIA's PTX, DFX captures computation and data movement semantics
 * without binding to specific hardware resources.
 *
 * DFX = Domain Flow Execution
 * - "Domain Flow" refers to the data flow patterns in domain-specific computations
 * - "Execution" emphasizes the executable nature of this representation
 *
 * Key Design Principles:
 * 1. Hardware-agnostic: Same DFX works on different KPU configurations
 * 2. Expressive: Captures all necessary scheduling decisions
 * 3. Optimizable: Allows driver-level optimization
 * 4. Serializable: Can be saved to disk and loaded later
 *
 * The driver/loader is responsible for:
 * - Concrete memory address assignment
 * - DMA engine/BlockMover/Streamer binding
 * - L1/L2/L3 tile allocation
 * - Cycle-accurate scheduling
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <memory>

namespace sw::kpu::compiler::dfx {

// ============================================================================
// Version and Identification
// ============================================================================

constexpr uint32_t DFX_VERSION_MAJOR = 1;
constexpr uint32_t DFX_VERSION_MINOR = 0;
constexpr uint32_t DFX_VERSION_PATCH = 0;

inline std::string dfx_version_string() {
    return std::to_string(DFX_VERSION_MAJOR) + "." +
           std::to_string(DFX_VERSION_MINOR) + "." +
           std::to_string(DFX_VERSION_PATCH);
}

// ============================================================================
// Data Types
// ============================================================================

/**
 * @brief Element data types supported by DFX
 */
enum class DataType {
    FLOAT32,    ///< 32-bit floating point
    FLOAT16,    ///< 16-bit floating point
    BFLOAT16,   ///< Brain floating point 16
    INT32,      ///< 32-bit signed integer
    INT16,      ///< 16-bit signed integer
    INT8,       ///< 8-bit signed integer
    UINT8,      ///< 8-bit unsigned integer
    BOOL        ///< Boolean type
};

/**
 * @brief Get size in bytes for a data type
 */
inline size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
        case DataType::INT32:
            return 4;
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
        case DataType::INT16:
            return 2;
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::BOOL:
            return 1;
    }
    return 0;
}

/**
 * @brief Convert data type to string
 */
inline std::string dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "f32";
        case DataType::FLOAT16: return "f16";
        case DataType::BFLOAT16: return "bf16";
        case DataType::INT32: return "i32";
        case DataType::INT16: return "i16";
        case DataType::INT8: return "i8";
        case DataType::UINT8: return "u8";
        case DataType::BOOL: return "bool";
    }
    return "unknown";
}

// ============================================================================
// Memory Hierarchy Levels
// ============================================================================

/**
 * @brief Abstract memory levels in the KPU hierarchy
 *
 * These are logical levels that get mapped to physical resources by the driver.
 */
enum class MemoryLevel {
    EXTERNAL,       ///< External DRAM (GDDR6)
    L3,             ///< L3 cache tiles
    L2,             ///< L2 cache banks
    L1,             ///< L1 streaming buffers
    REGISTER        ///< Systolic array registers (for accumulation)
};

inline std::string memory_level_to_string(MemoryLevel level) {
    switch (level) {
        case MemoryLevel::EXTERNAL: return "EXTERNAL";
        case MemoryLevel::L3: return "L3";
        case MemoryLevel::L2: return "L2";
        case MemoryLevel::L1: return "L1";
        case MemoryLevel::REGISTER: return "REGISTER";
    }
    return "UNKNOWN";
}

// ============================================================================
// Dataflow Strategies
// ============================================================================

/**
 * @brief Systolic array dataflow strategy
 *
 * Determines which operand stays stationary in the systolic array.
 */
enum class DataflowStrategy {
    OUTPUT_STATIONARY,  ///< C tiles stay in PEs, stream A and B
    WEIGHT_STATIONARY,  ///< B (weights) stay in PEs, stream A and accumulate C
    INPUT_STATIONARY    ///< A (inputs) stay in PEs, stream B and accumulate C
};

inline std::string dataflow_to_string(DataflowStrategy strategy) {
    switch (strategy) {
        case DataflowStrategy::OUTPUT_STATIONARY: return "output_stationary";
        case DataflowStrategy::WEIGHT_STATIONARY: return "weight_stationary";
        case DataflowStrategy::INPUT_STATIONARY: return "input_stationary";
    }
    return "unknown";
}

// ============================================================================
// Tensor Descriptors
// ============================================================================

/**
 * @brief Describes a tensor in the computation
 *
 * Tensors are named data arrays with shape and type information.
 */
struct TensorDescriptor {
    std::string name;           ///< Unique tensor name (e.g., "A", "B", "C")
    std::vector<size_t> shape;  ///< Tensor dimensions
    DataType dtype;             ///< Element data type
    bool is_constant;           ///< True if this tensor is read-only (e.g., weights)
    bool is_output;             ///< True if this tensor is a computation output

    /**
     * @brief Calculate total number of elements
     */
    size_t num_elements() const {
        size_t total = 1;
        for (auto dim : shape) {
            total *= dim;
        }
        return total;
    }

    /**
     * @brief Calculate total size in bytes
     */
    size_t size_bytes() const {
        return num_elements() * dtype_size(dtype);
    }
};

// ============================================================================
// Tile Specification
// ============================================================================

/**
 * @brief Specifies a tile within a tensor
 *
 * Tiles are sub-regions of tensors that can be moved through the memory
 * hierarchy and processed by the compute fabric.
 */
struct TileSpec {
    std::string tensor_name;            ///< Which tensor this tile belongs to
    MemoryLevel level;                  ///< Memory level where tile resides
    std::vector<size_t> tile_indices;   ///< Tile position in tile grid (e.g., [ti, tj])
    std::vector<size_t> tile_shape;     ///< Actual tile dimensions in elements

    /**
     * @brief Calculate tile size in bytes
     */
    size_t size_bytes(DataType dtype) const {
        size_t total = 1;
        for (auto dim : tile_shape) {
            total *= dim;
        }
        return total * dtype_size(dtype);
    }
};

// ============================================================================
// Operations
// ============================================================================

/**
 * @brief Base class for all DFX operations
 */
struct Operation {
    uint64_t op_id;                     ///< Unique operation identifier
    std::vector<uint64_t> depends_on;   ///< Operations this depends on (DAG edges)
    std::string label;                  ///< Optional human-readable label

    Operation() : op_id(0) {}
    virtual ~Operation() = default;

    /**
     * @brief Get operation type name for serialization
     */
    virtual std::string type_name() const = 0;
};

// ============================================================================
// Data Movement Operations
// ============================================================================

/**
 * @brief Types of data movement operations
 */
enum class DataMoveType {
    LOAD,       ///< Move data from lower to higher level (e.g., DRAM → L3)
    STORE,      ///< Move data from higher to lower level (e.g., L1 → L2)
    PREFETCH,   ///< Speculative load for latency hiding
    FLUSH       ///< Force writeback to lower level
};

inline std::string data_move_type_to_string(DataMoveType type) {
    switch (type) {
        case DataMoveType::LOAD: return "LOAD";
        case DataMoveType::STORE: return "STORE";
        case DataMoveType::PREFETCH: return "PREFETCH";
        case DataMoveType::FLUSH: return "FLUSH";
    }
    return "UNKNOWN";
}

/**
 * @brief Data movement operation
 *
 * Represents moving a tile between memory levels. The driver will
 * map this to specific DMA engine, BlockMover, or Streamer operations.
 */
struct DataMoveOp : public Operation {
    DataMoveType move_type;     ///< Type of data movement
    TileSpec source;            ///< Source tile specification
    TileSpec destination;       ///< Destination tile specification

    // Transform hints (for BlockMover)
    bool transpose;             ///< Apply transpose during move
    bool broadcast;             ///< Broadcast to multiple destinations

    DataMoveOp() : move_type(DataMoveType::LOAD), transpose(false), broadcast(false) {}

    std::string type_name() const override { return "DATA_MOVE"; }
};

// ============================================================================
// Compute Operations
// ============================================================================

/**
 * @brief Types of compute operations
 */
enum class ComputeType {
    MATMUL_TILE,        ///< Tile matrix multiplication (A × B → C)
    CONV2D_TILE,        ///< Tile 2D convolution
    ELEMENTWISE_ADD,    ///< Element-wise addition
    ELEMENTWISE_MUL,    ///< Element-wise multiplication
    ELEMENTWISE_SUB,    ///< Element-wise subtraction
    RELU,               ///< ReLU activation
    GELU,               ///< GELU activation
    SIGMOID,            ///< Sigmoid activation
    TANH,               ///< Tanh activation
    SOFTMAX_TILE,       ///< Tile softmax (along last dimension)
    LAYERNORM_TILE,     ///< Tile layer normalization
    REDUCE_SUM,         ///< Sum reduction
    REDUCE_MAX,         ///< Max reduction
    REDUCE_MEAN         ///< Mean reduction
};

inline std::string compute_type_to_string(ComputeType type) {
    switch (type) {
        case ComputeType::MATMUL_TILE: return "MATMUL_TILE";
        case ComputeType::CONV2D_TILE: return "CONV2D_TILE";
        case ComputeType::ELEMENTWISE_ADD: return "ELEMENTWISE_ADD";
        case ComputeType::ELEMENTWISE_MUL: return "ELEMENTWISE_MUL";
        case ComputeType::ELEMENTWISE_SUB: return "ELEMENTWISE_SUB";
        case ComputeType::RELU: return "RELU";
        case ComputeType::GELU: return "GELU";
        case ComputeType::SIGMOID: return "SIGMOID";
        case ComputeType::TANH: return "TANH";
        case ComputeType::SOFTMAX_TILE: return "SOFTMAX_TILE";
        case ComputeType::LAYERNORM_TILE: return "LAYERNORM_TILE";
        case ComputeType::REDUCE_SUM: return "REDUCE_SUM";
        case ComputeType::REDUCE_MAX: return "REDUCE_MAX";
        case ComputeType::REDUCE_MEAN: return "REDUCE_MEAN";
    }
    return "UNKNOWN";
}

/**
 * @brief Compute operation on tiles
 *
 * Represents a computation that consumes input tiles and produces output tiles.
 * The compute fabric (systolic array) executes these operations.
 */
struct ComputeOp : public Operation {
    ComputeType compute_type;           ///< Type of computation
    std::vector<TileSpec> inputs;       ///< Input tiles (in L1 or registers)
    TileSpec output;                    ///< Output tile (in L1 or registers)

    // For MATMUL: accumulation control
    bool accumulate;                    ///< Add to existing output vs overwrite

    // For reductions: reduction axis
    std::optional<int> reduction_axis;

    ComputeOp() : compute_type(ComputeType::MATMUL_TILE), accumulate(false) {}

    std::string type_name() const override { return "COMPUTE"; }
};

// ============================================================================
// Synchronization Operations
// ============================================================================

/**
 * @brief Synchronization barrier
 *
 * Represents a point where execution must wait for specified operations
 * to complete before proceeding.
 */
struct BarrierOp : public Operation {
    std::vector<uint64_t> wait_for;     ///< Wait for these operations to complete

    std::string type_name() const override { return "BARRIER"; }
};

// ============================================================================
// Loop Constructs
// ============================================================================

/**
 * @brief Tile iteration loop specification
 *
 * Represents a loop over tiles in one dimension. Multiple TileLoops
 * can be nested to iterate over the tile grid.
 */
struct TileLoop {
    std::string induction_var;          ///< Variable name (e.g., "ti", "tj", "tk")
    size_t start;                       ///< Starting tile index
    size_t end;                         ///< Ending tile index (exclusive)
    size_t step;                        ///< Step size (usually 1)

    // Pre-computed iteration order (optional)
    // If empty, iterate linearly from start to end
    std::vector<size_t> iteration_order;

    size_t num_iterations() const {
        return (end - start + step - 1) / step;
    }
};

// ============================================================================
// Tiling Configuration
// ============================================================================

/**
 * @brief Tiling parameters for a matrix operation
 *
 * For C[M,N] = A[M,K] × B[K,N]:
 * - Ti: tile size in M dimension
 * - Tj: tile size in N dimension
 * - Tk: tile size in K dimension
 */
struct TilingConfig {
    size_t Ti;          ///< M-dimension tile size
    size_t Tj;          ///< N-dimension tile size
    size_t Tk;          ///< K-dimension tile size

    // Optional L1 streaming configuration
    size_t L1_Ki;       ///< K-chunk for L1 streaming (sub-tile)

    // Derived values (computed from matrix dimensions)
    size_t num_tiles_m; ///< Number of tiles in M dimension
    size_t num_tiles_n; ///< Number of tiles in N dimension
    size_t num_tiles_k; ///< Number of tiles in K dimension

    TilingConfig() : Ti(0), Tj(0), Tk(0), L1_Ki(0),
                     num_tiles_m(0), num_tiles_n(0), num_tiles_k(0) {}
};

// ============================================================================
// Performance Hints
// ============================================================================

/**
 * @brief Performance hints generated by the compiler
 *
 * These hints help the driver make informed scheduling decisions
 * but are not binding constraints.
 */
struct PerformanceHints {
    size_t estimated_dram_bytes;        ///< Estimated DRAM traffic
    size_t estimated_compute_cycles;    ///< Estimated compute cycles
    double arithmetic_intensity;        ///< FLOPs per byte from DRAM
    size_t parallelism_degree;          ///< Amount of parallelism available

    // Tile reuse estimates
    size_t reuse_factor_a;              ///< How many times each A tile is reused
    size_t reuse_factor_b;              ///< How many times each B tile is reused

    PerformanceHints()
        : estimated_dram_bytes(0), estimated_compute_cycles(0),
          arithmetic_intensity(0.0), parallelism_degree(1),
          reuse_factor_a(1), reuse_factor_b(1) {}
};

// ============================================================================
// DFX Program
// ============================================================================

/**
 * @brief A complete DFX program representing a compiled kernel
 *
 * Contains all information needed for the driver to execute the kernel
 * on any compatible KPU configuration.
 */
struct Program {
    // Metadata
    std::string name;                   ///< Program/kernel name
    std::string source_graph;           ///< Original DFG file name
    uint32_t version_major;             ///< DFX version major
    uint32_t version_minor;             ///< DFX version minor
    uint32_t version_patch;             ///< DFX version patch

    // Configuration
    DataflowStrategy dataflow;          ///< Dataflow strategy
    TilingConfig tiling;                ///< Tiling configuration

    // Tensors
    std::vector<TensorDescriptor> tensors;

    // Operations (in dependency order)
    std::vector<std::unique_ptr<Operation>> operations;

    // Tile iteration loops (for nested loop generation)
    std::vector<TileLoop> tile_loops;

    // Performance hints
    PerformanceHints hints;

    Program()
        : version_major(DFX_VERSION_MAJOR),
          version_minor(DFX_VERSION_MINOR),
          version_patch(DFX_VERSION_PATCH),
          dataflow(DataflowStrategy::OUTPUT_STATIONARY) {}

    /**
     * @brief Get next available operation ID
     */
    uint64_t next_op_id() const {
        return operations.empty() ? 1 : operations.back()->op_id + 1;
    }

    /**
     * @brief Add an operation to the program
     */
    template<typename OpType>
    OpType& add_operation() {
        auto op = std::make_unique<OpType>();
        op->op_id = next_op_id();
        auto& ref = *op;
        operations.push_back(std::move(op));
        return ref;
    }

    /**
     * @brief Find tensor by name
     */
    const TensorDescriptor* find_tensor(const std::string& name) const {
        for (const auto& tensor : tensors) {
            if (tensor.name == name) {
                return &tensor;
            }
        }
        return nullptr;
    }
};

} // namespace sw::kpu::compiler::dfx
