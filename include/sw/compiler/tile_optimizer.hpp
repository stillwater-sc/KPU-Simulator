#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

namespace sw::kpu::compiler {

// Type aliases for clarity
using Size = size_t;
using Address = uint64_t;

/**
 * @brief Tile size optimizer for matrix multiplication on output stationary systolic arrays
 *
 * This optimizer determines optimal tile sizes for matrix multiplication C = A × B
 * where C[M,N] = A[M,K] × B[K,N], targeting the KPU memory hierarchy:
 *
 * Host Memory → External Memory → L3 Tiles → L2 Banks → L1 Buffers → Systolic Array
 *
 * Key Objectives:
 * 1. Minimize DRAM accesses by maximizing data reuse in L2/L3 caches
 * 2. Fit tiles within cache capacity constraints
 * 3. Align tiles to systolic array dimensions (16×16 for KPU)
 * 4. Maximize arithmetic intensity (FLOPs per byte transferred)
 *
 * Algorithm based on:
 * - Goto & van de Geijn (2008): Analytical tile size formulas
 * - Pouchet et al. (2012): Bounded search space reduction
 * - Output stationary dataflow optimization
 */
class TileOptimizer {
public:
    /**
     * @brief Configuration for a specific tiling strategy
     */
    struct TileConfig {
        // Tile dimensions (in elements, not bytes)
        Size Ti;  ///< M-dimension tile size (rows of A, rows of C)
        Size Tj;  ///< N-dimension tile size (cols of B, cols of C)
        Size Tk;  ///< K-dimension tile size (cols of A, rows of B)

        // L1 streaming configuration
        Size L1_Ki;  ///< K-chunk size for L1 streaming (typically 16-256)

        // Reuse factors
        Size reuse_A;  ///< How many times each A tile is reused
        Size reuse_B;  ///< How many times each B tile is reused
        Size reuse_C;  ///< K-accumulation factor (how many partial sums)

        // Cost metrics
        Size dram_accesses;     ///< Estimated DRAM accesses (bytes)
        Size l3_accesses;       ///< Estimated L3 accesses (bytes)
        Size l2_accesses;       ///< Estimated L2 accesses (bytes)
        double arithmetic_intensity;  ///< FLOPs per byte transferred from DRAM

        // Cache occupancy
        Size l2_footprint;  ///< Memory footprint in L2 (bytes)
        Size l3_footprint;  ///< Memory footprint in L3 (bytes)

        // Validation
        bool valid;         ///< Whether this configuration is valid
        std::string reason; ///< Reason if invalid

        TileConfig()
            : Ti(0), Tj(0), Tk(0), L1_Ki(0),
              reuse_A(0), reuse_B(0), reuse_C(0),
              dram_accesses(0), l3_accesses(0), l2_accesses(0),
              arithmetic_intensity(0.0),
              l2_footprint(0), l3_footprint(0),
              valid(false), reason("Not initialized") {}
    };

    /**
     * @brief Memory hierarchy specification for the target hardware
     */
    struct MemoryHierarchy {
        // Cache sizes (in bytes)
        Size L1_size;           ///< L1 buffer size per buffer
        Size L2_size;           ///< L2 bank size per bank
        Size L3_size;           ///< L3 tile size per tile
        Size external_size;     ///< External memory size

        // Cache counts
        Size L1_buffer_count;   ///< Number of L1 buffers (default: 4)
        Size L2_bank_count;     ///< Number of L2 banks (default: 8)
        Size L3_tile_count;     ///< Number of L3 tiles (default: 4)

        // Systolic array dimensions
        Size systolic_rows;     ///< Number of PE rows (default: 16)
        Size systolic_cols;     ///< Number of PE cols (default: 16)

        // Data type size
        Size element_size;      ///< Size of each element in bytes (default: 4 for float32)

        // Bandwidth (GB/s)
        double L1_bandwidth;
        double L2_bandwidth;
        double L3_bandwidth;
        double dram_bandwidth;

        /**
         * @brief Default KPU memory hierarchy
         */
        MemoryHierarchy()
            : L1_size(32 * 1024),           // 32 KB per L1 buffer
              L2_size(64 * 1024),           // 64 KB per L2 bank
              L3_size(128 * 1024),          // 128 KB per L3 tile
              external_size(1024 * 1024 * 1024),  // 1 GB
              L1_buffer_count(4),
              L2_bank_count(8),
              L3_tile_count(4),
              systolic_rows(16),
              systolic_cols(16),
              element_size(4),              // float32
              L1_bandwidth(1000.0),         // Very high (on-chip)
              L2_bandwidth(500.0),
              L3_bandwidth(250.0),
              dram_bandwidth(100.0) {}     // 100 GB/s
    };

    /**
     * @brief Search space bounds for tile sizes
     */
    struct SearchSpace {
        Size Ti_min, Ti_max;
        Size Tj_min, Tj_max;
        Size Tk_min, Tk_max;
        Size step;  ///< Step size (must align to systolic dimensions)

        SearchSpace() : Ti_min(0), Ti_max(0), Tj_min(0), Tj_max(0),
                       Tk_min(0), Tk_max(0), step(0) {}
    };

    /**
     * @brief Optimization strategy
     */
    enum class Strategy {
        ANALYTICAL,           ///< Fast analytical formula (Goto & van de Geijn style)
        BOUNDED_SEARCH,       ///< Exhaustive search within analytical bounds
        ML_PREDICTION,        ///< Machine learning prediction (future)
        HEURISTIC_HYBRID      ///< Hybrid approach (analytical + refinement)
    };

    /**
     * @brief Constructor
     * @param mem Memory hierarchy specification
     */
    explicit TileOptimizer(const MemoryHierarchy& mem = MemoryHierarchy())
        : memory_(mem) {}

    /**
     * @brief Main optimization API - select optimal tile sizes
     *
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C
     * @param K Number of columns in A, rows in B (reduction dimension)
     * @param strategy Optimization strategy to use
     * @return Optimal tile configuration
     */
    TileConfig optimize(Size M, Size N, Size K,
                       Strategy strategy = Strategy::ANALYTICAL);

    /**
     * @brief Fast analytical tile size calculation
     *
     * Based on Goto & van de Geijn (2008) formulas, adapted for output stationary.
     * Assumes:
     * - C tiles stay in systolic array PEs (output stationary)
     * - A and B tiles fit in target cache level
     * - Square tiles for initial estimate
     *
     * @param M, N, K Matrix dimensions
     * @param cache_size Target cache size (typically L2)
     * @return Tile configuration
     */
    TileConfig analytical_tiles(Size M, Size N, Size K, Size cache_size);

    /**
     * @brief Exhaustive search within bounded space
     *
     * Uses analytical bounds to dramatically reduce search space
     * (1,307× to 11,879× reduction according to Pouchet et al.)
     *
     * @param M, N, K Matrix dimensions
     * @param cache_size Target cache size
     * @return Optimal tile configuration within bounds
     */
    TileConfig search_tiles(Size M, Size N, Size K, Size cache_size);

    /**
     * @brief Heuristic hybrid approach
     *
     * Combines analytical solution with local refinement search
     *
     * @param M, N, K Matrix dimensions
     * @return Refined tile configuration
     */
    TileConfig heuristic_tiles(Size M, Size N, Size K);

    /**
     * @brief Calculate search space bounds using analytical models
     *
     * Conservative bound: Ti×Tk + Tk×Tj + Ti×Tj ≤ cache_size
     * Aggressive bound: Ti×Tk + Tk×Tj ≤ cache_size (assumes C stays in PEs)
     *
     * @param M, N, K Matrix dimensions
     * @param cache_size Target cache size
     * @return Search space bounds
     */
    SearchSpace calculate_bounds(Size M, Size N, Size K, Size cache_size);

    /**
     * @brief Estimate DRAM accesses for a given tile configuration
     *
     * Accounts for:
     * - A tile reuse across N dimension
     * - B tile reuse across M dimension
     * - C tile accumulation across K dimension (output stationary)
     * - L3 cache capacity for tile row/column storage
     *
     * @param M, N, K Matrix dimensions
     * @param config Tile configuration
     * @return Estimated DRAM accesses in bytes
     */
    Size estimate_dram_accesses(Size M, Size N, Size K, const TileConfig& config);

    /**
     * @brief Estimate total memory traffic at all hierarchy levels
     *
     * @param M, N, K Matrix dimensions
     * @param config Tile configuration
     * @return Updated config with all access counts
     */
    TileConfig estimate_memory_traffic(Size M, Size N, Size K, TileConfig config);

    /**
     * @brief Calculate reuse factors for a tile configuration
     *
     * @param M, N, K Matrix dimensions
     * @param config Tile configuration (will be updated)
     */
    void calculate_reuse_factors(Size M, Size N, Size K, TileConfig& config);

    /**
     * @brief Validate tile configuration against constraints
     *
     * Checks:
     * - Cache capacity constraints
     * - Systolic array alignment
     * - Minimum tile size requirements
     * - Memory hierarchy limits
     *
     * @param config Configuration to validate
     * @return true if valid, false otherwise (config.reason explains why)
     */
    bool validate(TileConfig& config);

    /**
     * @brief Get memory hierarchy specification
     */
    const MemoryHierarchy& memory_hierarchy() const { return memory_; }

    /**
     * @brief Set memory hierarchy specification
     */
    void set_memory_hierarchy(const MemoryHierarchy& mem) { memory_ = mem; }

private:
    MemoryHierarchy memory_;  ///< Memory hierarchy specification

    /**
     * @brief Round value down to nearest multiple
     */
    static Size round_down_to_multiple(Size value, Size multiple) {
        return (value / multiple) * multiple;
    }

    /**
     * @brief Round value up to nearest multiple
     */
    static Size round_up_to_multiple(Size value, Size multiple) {
        return ((value + multiple - 1) / multiple) * multiple;
    }

    /**
     * @brief Ceiling division
     */
    static Size ceil_div(Size numerator, Size denominator) {
        return (numerator + denominator - 1) / denominator;
    }

    /**
     * @brief Calculate cache footprint for a tile configuration
     *
     * For output stationary:
     * - A tile: Ti × Tk elements
     * - B tile: Tk × Tj elements
     * - C tile: Ti × Tj elements (may stay in PEs)
     *
     * @param Ti, Tj, Tk Tile dimensions
     * @param include_C Whether to include C in footprint
     * @return Footprint in bytes
     */
    Size calculate_footprint(Size Ti, Size Tj, Size Tk, bool include_C = true) const {
        Size footprint = (Ti * Tk + Tk * Tj) * memory_.element_size;
        if (include_C) {
            footprint += Ti * Tj * memory_.element_size;
        }
        return footprint;
    }

    /**
     * @brief Check if tile configuration fits in cache level
     */
    bool fits_in_cache(Size Ti, Size Tj, Size Tk, Size cache_size, bool include_C = true) const {
        return calculate_footprint(Ti, Tj, Tk, include_C) <= cache_size;
    }

    /**
     * @brief Evaluate a single tile configuration
     *
     * @param M, N, K Matrix dimensions
     * @param Ti, Tj, Tk Tile sizes
     * @return Complete tile configuration with metrics
     */
    TileConfig evaluate_config(Size M, Size N, Size K, Size Ti, Size Tj, Size Tk);
};

} // namespace sw::kpu::compiler
