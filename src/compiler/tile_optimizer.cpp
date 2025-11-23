#include <sw/compiler/tile_optimizer.hpp>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>

namespace sw::kpu::compiler {

TileOptimizer::TileConfig TileOptimizer::optimize(Size M, Size N, Size K, Strategy strategy) {
    switch (strategy) {
        case Strategy::ANALYTICAL:
            return analytical_tiles(M, N, K, memory_.L2_size);

        case Strategy::BOUNDED_SEARCH:
            return search_tiles(M, N, K, memory_.L2_size);

        case Strategy::HEURISTIC_HYBRID:
            return heuristic_tiles(M, N, K);

        case Strategy::ML_PREDICTION:
            // TODO: Implement ML-based prediction
            // For now, fall back to analytical
            return analytical_tiles(M, N, K, memory_.L2_size);

        default:
            return analytical_tiles(M, N, K, memory_.L2_size);
    }
}

TileOptimizer::TileConfig TileOptimizer::analytical_tiles(Size M, Size N, Size K, Size cache_size) {
    // Goto & van de Geijn (2008) style analytical formulas
    // Adapted for output stationary dataflow

    const Size sys_dim = memory_.systolic_rows;  // Assume square systolic array
    const Size elem_size = memory_.element_size;

    // Calculate optimal tile sizes using analytical formula
    // For cache size C and matrix dimensions M, N, K:
    //
    // Ti = min(M, sqrt(C × M / (2K + M)))
    // Tj = min(N, sqrt(C × N / (2K + N)))
    // Tk = min(K, (C - Ti×Tj) / (Ti + Tj))
    //
    // Then round to systolic boundary

    // Cache size in elements
    Size cache_elems = cache_size / elem_size;

    // Calculate ideal Ti
    double Ti_ideal = std::sqrt(static_cast<double>(cache_elems) * M / (2.0 * K + M));
    Size Ti = std::min(M, static_cast<Size>(Ti_ideal));
    Ti = round_down_to_multiple(Ti, sys_dim);
    Ti = std::max(Ti, sys_dim);  // At least one systolic tile

    // Calculate ideal Tj
    double Tj_ideal = std::sqrt(static_cast<double>(cache_elems) * N / (2.0 * K + N));
    Size Tj = std::min(N, static_cast<Size>(Tj_ideal));
    Tj = round_down_to_multiple(Tj, sys_dim);
    Tj = std::max(Tj, sys_dim);

    // Calculate Tk based on remaining cache capacity
    // For output stationary, C stays in PEs, so we have more room for A and B
    Size remaining = cache_elems;
    if (Ti * Tj <= remaining) {
        remaining -= Ti * Tj;  // Reserve space for C if needed
    }

    Size Tk_ideal = remaining / (Ti + Tj);
    Size Tk = std::min(K, Tk_ideal);
    Tk = round_down_to_multiple(Tk, sys_dim);
    Tk = std::max(Tk, sys_dim);

    // Verify cache constraint and adjust if needed
    while (!fits_in_cache(Ti, Tj, Tk, cache_size, true)) {
        // Reduce Tk first (least impact on reuse)
        if (Tk > sys_dim) {
            Tk -= sys_dim;
        } else if (Tj > sys_dim) {
            // Reduce Tj next
            Tj -= sys_dim;
        } else if (Ti > sys_dim) {
            // Reduce Ti last
            Ti -= sys_dim;
        } else {
            // Can't reduce further
            break;
        }
    }

    // L1 K-chunk size for streaming
    Size L1_Ki = std::min(Tk, memory_.L1_size / (2 * sys_dim * elem_size));
    L1_Ki = round_down_to_multiple(L1_Ki, sys_dim);
    L1_Ki = std::max(L1_Ki, sys_dim);

    // Create configuration and evaluate
    return evaluate_config(M, N, K, Ti, Tj, Tk);
}

TileOptimizer::TileConfig TileOptimizer::search_tiles(Size M, Size N, Size K, Size cache_size) {
    // Calculate search space bounds
    SearchSpace space = calculate_bounds(M, N, K, cache_size);

    TileConfig best_config;
    Size min_dram_accesses = std::numeric_limits<Size>::max();

    // Exhaustive search within bounded space
    for (Size Ti = space.Ti_min; Ti <= space.Ti_max; Ti += space.step) {
        for (Size Tj = space.Tj_min; Tj <= space.Tj_max; Tj += space.step) {
            for (Size Tk = space.Tk_min; Tk <= space.Tk_max; Tk += space.step) {
                // Check cache constraint
                if (!fits_in_cache(Ti, Tj, Tk, cache_size, true)) {
                    continue;
                }

                // Evaluate this configuration
                TileConfig config = evaluate_config(M, N, K, Ti, Tj, Tk);

                if (config.valid && config.dram_accesses < min_dram_accesses) {
                    min_dram_accesses = config.dram_accesses;
                    best_config = config;
                }
            }
        }
    }

    if (!best_config.valid) {
        // Fall back to analytical if search failed
        best_config = analytical_tiles(M, N, K, cache_size);
    }

    return best_config;
}

TileOptimizer::TileConfig TileOptimizer::heuristic_tiles(Size M, Size N, Size K) {
    // Start with analytical solution
    TileConfig config = analytical_tiles(M, N, K, memory_.L2_size);

    if (!config.valid) {
        return config;
    }

    // Try local refinement: test neighboring tile sizes
    const Size sys_dim = memory_.systolic_rows;
    Size best_dram = config.dram_accesses;
    TileConfig best_config = config;

    // Test Ti ± systolic_dim
    for (int delta_i : {-1, 0, 1}) {
        for (int delta_j : {-1, 0, 1}) {
            for (int delta_k : {-1, 0, 1}) {
                Size Ti = config.Ti + delta_i * sys_dim;
                Size Tj = config.Tj + delta_j * sys_dim;
                Size Tk = config.Tk + delta_k * sys_dim;

                // Bounds check
                if (Ti < sys_dim || Tj < sys_dim || Tk < sys_dim) continue;
                if (Ti > M || Tj > N || Tk > K) continue;

                // Cache constraint check
                if (!fits_in_cache(Ti, Tj, Tk, memory_.L2_size, true)) continue;

                TileConfig test_config = evaluate_config(M, N, K, Ti, Tj, Tk);

                if (test_config.valid && test_config.dram_accesses < best_dram) {
                    best_dram = test_config.dram_accesses;
                    best_config = test_config;
                }
            }
        }
    }

    return best_config;
}

TileOptimizer::SearchSpace TileOptimizer::calculate_bounds(Size M, Size N, Size K, Size cache_size) {
    SearchSpace space;
    const Size sys_dim = memory_.systolic_rows;
    const Size elem_size = memory_.element_size;

    // Minimum: At least one systolic tile
    space.Ti_min = sys_dim;
    space.Tj_min = sys_dim;
    space.Tk_min = sys_dim;
    space.step = sys_dim;

    // Maximum: Largest tile that fits in cache
    Size cache_elems = cache_size / elem_size;

    // Conservative upper bound: assume square tiles
    // 3 × tile^2 ≤ cache_elems (for Ti×Tk + Tk×Tj + Ti×Tj)
    Size max_tile_sq = std::sqrt(cache_elems / 3.0);
    max_tile_sq = round_down_to_multiple(max_tile_sq, sys_dim);

    space.Ti_max = std::min(M, max_tile_sq);
    space.Tj_max = std::min(N, max_tile_sq);

    // For Tk, we can be more aggressive since C stays in PEs
    // Ti×Tk + Tk×Tj ≤ cache_elems
    Size max_Tk = cache_elems / (space.Ti_max + space.Tj_max);
    max_Tk = round_down_to_multiple(max_Tk, sys_dim);
    space.Tk_max = std::min(K, max_Tk);

    // Ensure minimums don't exceed maximums
    space.Ti_max = std::max(space.Ti_max, space.Ti_min);
    space.Tj_max = std::max(space.Tj_max, space.Tj_min);
    space.Tk_max = std::max(space.Tk_max, space.Tk_min);

    return space;
}

Size TileOptimizer::estimate_dram_accesses(Size M, Size N, Size K, const TileConfig& config) {
    const Size elem_size = memory_.element_size;

    // Number of tiles in each dimension
    Size M_tiles = ceil_div(M, config.Ti);
    Size N_tiles = ceil_div(N, config.Tj);
    Size K_tiles = ceil_div(K, config.Tk);

    // Assume L3 can hold one row of A tiles and one column of B tiles
    // This is a simplification; actual reuse depends on L3 size
    // Size L3_capacity_tiles = memory_.L3_tile_count;  // Reserved for future use

    // A tile reuse: each A[i,k] tile is used for N_tiles different B tiles
    // But some may be cached in L3
    Size A_tile_size = config.Ti * config.Tk * elem_size;
    Size A_total_tiles = M_tiles * K_tiles;
    Size A_dram_fetches = A_total_tiles * A_tile_size / std::max(Size(1), config.reuse_A);

    // B tile reuse: each B[k,j] tile is used for M_tiles different A tiles
    Size B_tile_size = config.Tk * config.Tj * elem_size;
    Size B_total_tiles = K_tiles * N_tiles;
    Size B_dram_fetches = B_total_tiles * B_tile_size / std::max(Size(1), config.reuse_B);

    // C tile: Written once after K accumulations (output stationary advantage)
    Size C_tile_size = config.Ti * config.Tj * elem_size;
    Size C_total_tiles = M_tiles * N_tiles;
    Size C_dram_stores = C_total_tiles * C_tile_size;

    return A_dram_fetches + B_dram_fetches + C_dram_stores;
}

TileOptimizer::TileConfig TileOptimizer::estimate_memory_traffic(Size M, Size N, Size K, TileConfig config) {
    const Size elem_size = memory_.element_size;

    // Calculate tile counts
    Size M_tiles = ceil_div(M, config.Ti);
    Size N_tiles = ceil_div(N, config.Tj);
    Size K_tiles = ceil_div(K, config.Tk);

    // Tile sizes
    Size A_tile = config.Ti * config.Tk * elem_size;
    Size B_tile = config.Tk * config.Tj * elem_size;
    Size C_tile = config.Ti * config.Tj * elem_size;

    // DRAM accesses (with reuse)
    config.dram_accesses = estimate_dram_accesses(M, N, K, config);

    // L3 accesses: Every tile passes through L3
    config.l3_accesses = (M_tiles * K_tiles * A_tile) +
                         (K_tiles * N_tiles * B_tile) +
                         (M_tiles * N_tiles * C_tile);

    // L2 accesses: Tiles are reused at L2 level for systolic array tiles
    Size systolic_tiles_per_cache = config.Ti / memory_.systolic_rows;
    config.l2_accesses = config.l3_accesses * systolic_tiles_per_cache;

    // Arithmetic intensity: FLOPs per byte from DRAM
    Size total_flops = 2 * M * N * K;  // 2 ops per element (multiply-add)
    config.arithmetic_intensity = static_cast<double>(total_flops) / config.dram_accesses;

    return config;
}

void TileOptimizer::calculate_reuse_factors(Size M, Size N, Size K, TileConfig& config) {
    // A tile reuse: how many times A[i,k] is used
    // Each A tile is used for all B tiles in the N dimension
    config.reuse_A = ceil_div(N, config.Tj);

    // B tile reuse: how many times B[k,j] is used
    // Each B tile is used for all A tiles in the M dimension
    config.reuse_B = ceil_div(M, config.Ti);

    // C tile accumulation: how many K-chunks accumulate into C
    // This is the output stationary advantage
    config.reuse_C = ceil_div(K, config.Tk);
}

bool TileOptimizer::validate(TileConfig& config) {
    const Size sys_dim = memory_.systolic_rows;

    // Check alignment to systolic array dimensions
    if (config.Ti % sys_dim != 0) {
        config.valid = false;
        config.reason = "Ti not aligned to systolic dimensions";
        return false;
    }

    if (config.Tj % sys_dim != 0) {
        config.valid = false;
        config.reason = "Tj not aligned to systolic dimensions";
        return false;
    }

    if (config.Tk % sys_dim != 0) {
        config.valid = false;
        config.reason = "Tk not aligned to systolic dimensions";
        return false;
    }

    // Check minimum tile sizes
    if (config.Ti < sys_dim || config.Tj < sys_dim || config.Tk < sys_dim) {
        config.valid = false;
        config.reason = "Tile size below minimum (systolic dimension)";
        return false;
    }

    // Calculate cache footprints
    config.l2_footprint = calculate_footprint(config.Ti, config.Tj, config.Tk, true);
    config.l3_footprint = config.l2_footprint;  // Similar for now

    // Check L2 capacity constraint
    if (config.l2_footprint > memory_.L2_size) {
        config.valid = false;
        std::ostringstream oss;
        oss << "L2 footprint (" << config.l2_footprint << " bytes) exceeds L2 size ("
            << memory_.L2_size << " bytes)";
        config.reason = oss.str();
        return false;
    }

    // Check L1 constraints for streaming
    Size L1_requirement = (config.Ti + config.Tj) * config.L1_Ki * memory_.element_size;
    if (L1_requirement > memory_.L1_size * memory_.L1_buffer_count) {
        config.valid = false;
        config.reason = "L1 requirement exceeds available L1 buffers";
        return false;
    }

    config.valid = true;
    config.reason = "Valid configuration";
    return true;
}

TileOptimizer::TileConfig TileOptimizer::evaluate_config(Size M, Size N, Size K,
                                                          Size Ti, Size Tj, Size Tk) {
    TileConfig config;
    config.Ti = Ti;
    config.Tj = Tj;
    config.Tk = Tk;

    // L1 K-chunk size
    config.L1_Ki = std::min(Tk, memory_.L1_size / (2 * memory_.systolic_rows * memory_.element_size));
    config.L1_Ki = round_down_to_multiple(config.L1_Ki, memory_.systolic_rows);
    config.L1_Ki = std::max(config.L1_Ki, memory_.systolic_rows);

    // Calculate reuse factors
    calculate_reuse_factors(M, N, K, config);

    // Estimate memory traffic
    config = estimate_memory_traffic(M, N, K, config);

    // Validate
    validate(config);

    return config;
}

} // namespace sw::kpu::compiler
