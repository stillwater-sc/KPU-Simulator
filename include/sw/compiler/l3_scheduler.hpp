/**
 * @file l3_scheduler.hpp
 * @brief L3 cache scheduler and DRAM overfetch analysis
 *
 * This component sits above the L2 tile scheduler and models the L3 cache behavior.
 * It answers the critical question: "How much DRAM traffic do we actually generate?"
 *
 * Memory Hierarchy:
 *   DRAM → L3 → L2 → L1 (PE registers) → Compute
 *
 * The L3 scheduler:
 * 1. Takes an L2 schedule (sequence of L2 tile loads)
 * 2. Simulates which L2 tiles hit/miss in L3
 * 3. Calculates DRAM→L3 and L3→L2 traffic
 * 4. Quantifies overfetching factor for each tensor
 *
 * Key Insight:
 * - Small L3 → Tiles evicted and reloaded → High DRAM traffic (overfetching)
 * - Large L3 → Tiles stay resident → Low DRAM traffic (minimal overfetching)
 *
 * Overfetch Factor:
 *   overfetch = (actual bytes moved from DRAM) / (minimum bytes needed)
 *
 *   Example: 1024×1024 matrix = 4MB
 *   - Ideal: Load 4MB once → overfetch = 1.0×
 *   - Reality: Small L3 causes reload 3 times → 12MB → overfetch = 3.0×
 */

#pragma once

#include <sw/compiler/l2_tile_scheduler.hpp>
#include <sw/compiler/schedule_characterizer.hpp>
#include <sw/concepts.hpp>
#include <vector>
#include <map>
#include <string>
#include <memory>

namespace sw::kpu::compiler {

/**
 * @brief L3 cache configuration
 */
struct L3Config {
    Size l3_capacity;           ///< L3 cache size per tile (bytes)
    Size l3_tile_count;         ///< Number of L3 tiles in system
    Size l3_associativity;      ///< Set associativity (e.g., 8-way)
    Size cache_line_size;       ///< Cache line size (bytes, e.g., 64B)

    // Default: 16MB L3 per tile, 4 tiles
    L3Config()
        : l3_capacity(16 * 1024 * 1024)
        , l3_tile_count(4)
        , l3_associativity(8)
        , cache_line_size(64)
    {}

    Size total_l3_capacity() const {
        return l3_capacity * l3_tile_count;
    }
};

/**
 * @brief Reference to a tensor region (for tracking what's in L3)
 */
struct TensorRegion {
    enum TensorType { A, B, C };

    TensorType type;            ///< Which tensor (A, B, or C)
    Size offset;                ///< Offset in tensor (bytes)
    Size size;                  ///< Size of region (bytes)
    Size access_time;           ///< Last access time (for LRU)

    TensorRegion(TensorType t, Size off, Size sz, Size time)
        : type(t), offset(off), size(sz), access_time(time) {}

    bool operator<(const TensorRegion& other) const {
        if (type != other.type) return type < other.type;
        return offset < other.offset;
    }
};

/**
 * @brief L3 cache simulation results
 */
struct L3Schedule {
    // L3 configuration
    L3Config config;

    // Traffic analysis (in bytes)
    Size dram_to_l3_bytes;      ///< Total DRAM→L3 traffic
    Size l3_to_l2_bytes;        ///< Total L3→L2 traffic
    Size total_dram_reads;      ///< Total bytes read from DRAM
    Size total_dram_writes;     ///< Total bytes written to DRAM

    // Per-tensor overfetch analysis
    Size tensor_a_size;         ///< Ideal size of tensor A (bytes)
    Size tensor_b_size;         ///< Ideal size of tensor B (bytes)
    Size tensor_c_size;         ///< Ideal size of tensor C (bytes)

    Size tensor_a_fetched;      ///< Actual bytes fetched for A
    Size tensor_b_fetched;      ///< Actual bytes fetched for B
    Size tensor_c_fetched;      ///< Actual bytes fetched for C

    double overfetch_factor_a;  ///< A overfetch = fetched / ideal
    double overfetch_factor_b;  ///< B overfetch = fetched / ideal
    double overfetch_factor_c;  ///< C overfetch = fetched / ideal
    double overfetch_factor_total; ///< Overall overfetch factor

    // L3 utilization
    Size l3_capacity_used_peak; ///< Peak L3 usage (bytes)
    double l3_utilization;      ///< Peak usage / capacity
    Size num_l3_evictions;      ///< Number of cache evictions

    // Hit/miss statistics
    Size l2_tile_loads_total;   ///< Total L2 tile load operations
    Size l2_tile_loads_hit_l3;  ///< Loads that hit in L3
    Size l2_tile_loads_miss_l3; ///< Loads that miss in L3 (go to DRAM)
    double l3_hit_rate;         ///< Hit rate = hits / total

    L3Schedule()
        : dram_to_l3_bytes(0), l3_to_l2_bytes(0)
        , total_dram_reads(0), total_dram_writes(0)
        , tensor_a_size(0), tensor_b_size(0), tensor_c_size(0)
        , tensor_a_fetched(0), tensor_b_fetched(0), tensor_c_fetched(0)
        , overfetch_factor_a(1.0), overfetch_factor_b(1.0), overfetch_factor_c(1.0)
        , overfetch_factor_total(1.0)
        , l3_capacity_used_peak(0), l3_utilization(0)
        , num_l3_evictions(0)
        , l2_tile_loads_total(0), l2_tile_loads_hit_l3(0), l2_tile_loads_miss_l3(0)
        , l3_hit_rate(0)
    {}
};

/**
 * @brief L3 cache simulator (simple LRU model)
 *
 * This is a simplified cache model that tracks which tensor regions
 * are resident in L3 and simulates evictions using LRU policy.
 */
class L3CacheSimulator {
public:
    explicit L3CacheSimulator(const L3Config& config);

    /**
     * @brief Simulate loading a tensor region into L3
     *
     * @param region The tensor region to load
     * @return True if it was already in L3 (hit), false if it needed to be fetched from DRAM (miss)
     */
    bool load(const TensorRegion& region);

    /**
     * @brief Get current L3 occupancy
     */
    Size get_occupancy() const { return current_occupancy_; }

    /**
     * @brief Get peak L3 occupancy
     */
    Size get_peak_occupancy() const { return peak_occupancy_; }

    /**
     * @brief Get number of evictions
     */
    Size get_num_evictions() const { return num_evictions_; }

    /**
     * @brief Reset the cache state
     */
    void reset();

private:
    L3Config config_;
    std::vector<TensorRegion> cache_contents_;  ///< Regions currently in L3
    Size current_occupancy_;                     ///< Current bytes in L3
    Size peak_occupancy_;                        ///< Peak bytes in L3
    Size num_evictions_;                         ///< Count of evictions
    Size access_counter_;                        ///< For LRU tracking

    /**
     * @brief Evict regions using LRU until we have enough space
     */
    void evict_to_make_space(Size needed_space);

    /**
     * @brief Check if a region is in cache
     */
    bool is_cached(const TensorRegion& region) const;
};

/**
 * @brief L3 scheduler - models L3 cache behavior for an L2 schedule
 */
class L3Scheduler {
public:
    explicit L3Scheduler(const L3Config& config = L3Config());

    /**
     * @brief Generate L3 schedule from L2 schedule
     *
     * Simulates executing the L2 schedule and tracks:
     * - Which L2 tile loads hit in L3 vs go to DRAM
     * - DRAM traffic (bytes read/written)
     * - Overfetching factor per tensor
     *
     * @param shape Tensor shape (M×N×K)
     * @param l2_schedule The L2 tiling schedule
     * @param element_size Size of each element (bytes, default 4 for FP32)
     * @return L3 schedule with traffic and overfetch analysis
     */
    L3Schedule schedule_l3(
        const TensorShape& shape,
        const L2TileScheduler::L2Schedule& l2_schedule,
        Size element_size = 4);

    /**
     * @brief Analyze DRAM overfetching for different L3 sizes
     *
     * Sweeps L3 capacity and measures overfetch factor.
     * Useful for understanding L3 size vs DRAM traffic tradeoff.
     *
     * @param shape Tensor shape
     * @param l2_schedule L2 schedule
     * @param l3_sizes Vector of L3 sizes to test
     * @param element_size Element size in bytes
     * @return Map from L3 size to L3Schedule
     */
    std::map<Size, L3Schedule> sweep_l3_size(
        const TensorShape& shape,
        const L2TileScheduler::L2Schedule& l2_schedule,
        const std::vector<Size>& l3_sizes,
        Size element_size = 4);

    /**
     * @brief Calculate ideal (minimum) DRAM traffic
     *
     * This is the theoretical minimum: load each tensor exactly once.
     *
     * @param shape Tensor shape
     * @param element_size Element size in bytes
     * @return Minimum DRAM bytes needed
     */
    static Size calculate_ideal_dram_traffic(
        const TensorShape& shape,
        Size element_size = 4);

    // Accessors
    const L3Config& config() const { return config_; }

private:
    L3Config config_;

    /**
     * @brief Simulate L2 tile load sequence
     *
     * For each L2 tile load in the schedule:
     * 1. Check if tile data is in L3 (hit/miss)
     * 2. If miss, fetch from DRAM
     * 3. Update L3 contents (may evict other tiles)
     *
     * @param shape Tensor shape
     * @param l2_schedule L2 schedule to simulate
     * @param element_size Element size
     * @param simulator L3 cache simulator
     * @return L3 schedule with statistics
     */
    L3Schedule simulate_l2_execution(
        const TensorShape& shape,
        const L2TileScheduler::L2Schedule& l2_schedule,
        Size element_size,
        L3CacheSimulator& simulator);

    /**
     * @brief Get tensor region for an L2 tile
     *
     * Maps L2 tile coordinates to a tensor region (which tensor, offset, size).
     *
     * @param tensor_type Which tensor (A, B, or C)
     * @param tile_i Tile row index
     * @param tile_j Tile column index (for B and C)
     * @param tile_k Tile K index (for A and B)
     * @param tile_dims Tile dimensions (Ti, Tj, Tk)
     * @param element_size Element size
     * @return Tensor region representing this tile
     */
    TensorRegion get_tile_region(
        TensorRegion::TensorType tensor_type,
        Size tile_i, Size tile_j, Size tile_k,
        const std::tuple<Size, Size, Size>& tile_dims,
        Size element_size,
        Size access_time) const;
};

} // namespace sw::kpu::compiler
