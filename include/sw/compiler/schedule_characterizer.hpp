/**
 * @file schedule_characterizer.hpp
 * @brief Characterization framework for scheduling strategies across tensor workloads
 *
 * This framework evaluates the Pareto frontier of scheduling strategies (weight-stationary,
 * input-stationary, output-stationary) across hundreds of thousands of realistic tensor
 * shapes. It measures energy and latency compared to ideal execution to identify optimal
 * dataflow strategies for real-world workloads.
 *
 * Key Concepts:
 * - Ideal Performance: What systolic array could achieve with perfect tiling
 * - Actual Performance: What schedule achieves with real constraints
 * - Slowdown Factor: actual_cost / ideal_cost (lower is better)
 * - Pareto Frontier: Set of non-dominated (energy, latency) points
 */

#pragma once

#include <sw/compiler/tile_optimizer.hpp>
#include <sw/compiler/l2_tile_scheduler.hpp>
#include <sw/concepts.hpp>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <functional>

namespace sw::kpu::compiler {

/**
 * @brief Tensor shape for characterization
 */
struct TensorShape {
    Size M, N, K;  ///< Matrix dimensions for C[M,N] = A[M,K] × B[K,N]

    TensorShape(Size m = 0, Size n = 0, Size k = 0) : M(m), N(n), K(k) {}

    std::string to_string() const {
        return std::to_string(M) + "×" + std::to_string(N) + "×" + std::to_string(K);
    }

    bool operator<(const TensorShape& other) const {
        if (M != other.M) return M < other.M;
        if (N != other.N) return N < other.N;
        return K < other.K;
    }
};

/**
 * @brief Dataflow strategies for systolic arrays
 */
enum class DataflowStrategy {
    WEIGHT_STATIONARY,   ///< Weights stay in PEs, stream inputs/outputs
    INPUT_STATIONARY,    ///< Inputs stay in PEs, stream weights/outputs
    OUTPUT_STATIONARY    ///< Outputs accumulate in PEs, stream inputs/weights
};

/**
 * @brief Performance metrics for a schedule
 */
struct PerformanceMetrics {
    // Energy components (in arbitrary units or joules)
    double dram_energy;          ///< DRAM access energy
    double l3_energy;            ///< L3 cache energy
    double l2_energy;            ///< L2 cache energy
    double l1_energy;            ///< L1 buffer energy
    double compute_energy;       ///< Compute (MAC) energy
    double total_energy;         ///< Sum of all energy

    // Latency components (in cycles)
    Cycle dram_cycles;           ///< DRAM access cycles
    Cycle l3_cycles;             ///< L3 access cycles
    Cycle l2_cycles;             ///< L2 access cycles
    Cycle l1_cycles;             ///< L1 buffer cycles
    Cycle compute_cycles;        ///< Compute cycles
    Cycle total_cycles;          ///< Critical path cycles

    // Data movement (bytes)
    Size dram_accesses;          ///< Bytes from DRAM
    Size l3_accesses;            ///< Bytes from L3
    Size l2_accesses;            ///< Bytes from L2
    Size l1_accesses;            ///< Bytes from L1

    // Efficiency metrics
    double arithmetic_intensity;  ///< FLOPs per byte from DRAM
    double utilization;          ///< PE utilization (0-1)
    double bandwidth_efficiency; ///< Actual bandwidth / peak bandwidth

    // Reuse factors
    Size reuse_A;
    Size reuse_B;
    Size reuse_C;

    PerformanceMetrics()
        : dram_energy(0), l3_energy(0), l2_energy(0), l1_energy(0), compute_energy(0), total_energy(0)
        , dram_cycles(0), l3_cycles(0), l2_cycles(0), l1_cycles(0), compute_cycles(0), total_cycles(0)
        , dram_accesses(0), l3_accesses(0), l2_accesses(0), l1_accesses(0)
        , arithmetic_intensity(0), utilization(0), bandwidth_efficiency(0)
        , reuse_A(0), reuse_B(0), reuse_C(0) {}
};

/**
 * @brief Ideal performance (theoretical best case)
 */
struct IdealMetrics {
    Cycle ideal_cycles;          ///< Minimum cycles with perfect tiling
    double ideal_energy;         ///< Minimum energy with perfect data reuse
    double peak_throughput;      ///< Peak FLOPs/sec

    IdealMetrics() : ideal_cycles(0), ideal_energy(0), peak_throughput(0) {}
};

/**
 * @brief Schedule evaluation result
 */
struct ScheduleEvaluation {
    TensorShape shape;
    DataflowStrategy strategy;
    TileOptimizer::TileConfig tile_config;

    // Performance
    PerformanceMetrics metrics;
    IdealMetrics ideal;

    // Slowdown factors (actual / ideal)
    double energy_slowdown;      ///< total_energy / ideal_energy
    double latency_slowdown;     ///< total_cycles / ideal_cycles

    // Pareto properties
    bool is_pareto_optimal;      ///< Is this on Pareto frontier?
    size_t dominated_by_count;   ///< How many schedules dominate this?

    ScheduleEvaluation()
        : strategy(DataflowStrategy::OUTPUT_STATIONARY)
        , energy_slowdown(1.0), latency_slowdown(1.0)
        , is_pareto_optimal(false), dominated_by_count(0) {}

    std::string to_string() const {
        return shape.to_string() + " [" +
               dataflow_to_string(strategy) + "]";
    }

    static std::string dataflow_to_string(DataflowStrategy s) {
        switch (s) {
            case DataflowStrategy::WEIGHT_STATIONARY: return "WS";
            case DataflowStrategy::INPUT_STATIONARY: return "IS";
            case DataflowStrategy::OUTPUT_STATIONARY: return "OS";
            default: return "??";
        }
    }
};

/**
 * @brief Pareto point on energy-latency frontier
 */
struct ParetoPoint {
    double energy;
    Cycle latency;
    std::shared_ptr<ScheduleEvaluation> schedule;

    ParetoPoint() : energy(0), latency(0), schedule(nullptr) {}
    ParetoPoint(double e, Cycle l, std::shared_ptr<ScheduleEvaluation> s)
        : energy(e), latency(l), schedule(s) {}

    // For sorting by energy
    bool operator<(const ParetoPoint& other) const {
        if (energy != other.energy) return energy < other.energy;
        return latency < other.latency;
    }
};

/**
 * @brief Pareto frontier (non-dominated schedules)
 */
struct ParetoFrontier {
    std::vector<ParetoPoint> points;  ///< Sorted by energy

    // Statistics
    size_t total_schedules;           ///< Total schedules evaluated
    size_t frontier_size;             ///< Number of Pareto-optimal schedules
    double coverage_percentage;       ///< % of schedules on frontier

    ParetoFrontier() : total_schedules(0), frontier_size(0), coverage_percentage(0) {}
};

/**
 * @brief Energy model parameters
 */
struct EnergyModel {
    // Energy per access (pJ)
    double dram_read_pj;         ///< DRAM read energy
    double dram_write_pj;        ///< DRAM write energy
    double l3_read_pj;           ///< L3 read energy
    double l3_write_pj;          ///< L3 write energy
    double l2_read_pj;           ///< L2 read energy
    double l2_write_pj;          ///< L2 write energy
    double l1_read_pj;           ///< L1 read energy
    double l1_write_pj;          ///< L1 write energy
    double mac_pj;               ///< MAC operation energy

    // Default: Based on Eyeriss / TPU-like accelerators
    EnergyModel()
        : dram_read_pj(200.0)    // 200 pJ per byte
        , dram_write_pj(200.0)
        , l3_read_pj(10.0)       // 10 pJ per byte
        , l3_write_pj(10.0)
        , l2_read_pj(5.0)        // 5 pJ per byte
        , l2_write_pj(5.0)
        , l1_read_pj(1.0)        // 1 pJ per byte
        , l1_write_pj(1.0)
        , mac_pj(0.2)            // 0.2 pJ per MAC
    {}
};

/**
 * @brief Latency model parameters
 */
struct LatencyModel {
    // Cycles per access
    Cycle dram_latency;          ///< DRAM access latency (cycles)
    Cycle l3_latency;            ///< L3 access latency
    Cycle l2_latency;            ///< L2 access latency
    Cycle l1_latency;            ///< L1 access latency

    // Bandwidth (GB/s)
    double dram_bandwidth;
    double l3_bandwidth;
    double l2_bandwidth;
    double l1_bandwidth;

    // Compute
    double clock_freq_ghz;       ///< System clock frequency
    Cycle mac_latency;           ///< MAC operation latency

    // Default: Based on realistic accelerator parameters
    LatencyModel()
        : dram_latency(100)
        , l3_latency(20)
        , l2_latency(10)
        , l1_latency(1)
        , dram_bandwidth(100.0)
        , l3_bandwidth(400.0)
        , l2_bandwidth(800.0)
        , l1_bandwidth(1600.0)
        , clock_freq_ghz(1.0)
        , mac_latency(1)
    {}
};

/**
 * @brief Workload generator for characterization
 */
class WorkloadGenerator {
public:
    /**
     * @brief Generate realistic tensor shapes for ML workloads
     *
     * @param count Number of shapes to generate
     * @param distribution Distribution type (uniform, power-law, real-world)
     * @return Vector of tensor shapes
     */
    static std::vector<TensorShape> generate_ml_workloads(
        size_t count,
        const std::string& distribution = "real-world");

    /**
     * @brief Generate shapes covering full parameter space
     */
    static std::vector<TensorShape> generate_sweep(
        Size m_min, Size m_max, Size m_step,
        Size n_min, Size n_max, Size n_step,
        Size k_min, Size k_max, Size k_step);

    /**
     * @brief Generate common layer sizes from popular networks
     */
    static std::vector<TensorShape> generate_from_networks(
        const std::vector<std::string>& networks = {
            "resnet50", "vgg16", "bert", "gpt2", "mobilenet"
        });
};

/**
 * @brief Schedule characterizer
 */
class ScheduleCharacterizer {
public:
    ScheduleCharacterizer(
        const TileOptimizer::MemoryHierarchy& mem = TileOptimizer::MemoryHierarchy(),
        const EnergyModel& energy = EnergyModel(),
        const LatencyModel& latency = LatencyModel());

    /**
     * @brief Evaluate a single schedule for a tensor shape
     */
    ScheduleEvaluation evaluate_schedule(
        const TensorShape& shape,
        DataflowStrategy strategy);

    /**
     * @brief Evaluate all strategies for a tensor shape
     */
    std::vector<ScheduleEvaluation> evaluate_all_strategies(
        const TensorShape& shape);

    /**
     * @brief Characterize across many workloads
     */
    ParetoFrontier characterize_workloads(
        const std::vector<TensorShape>& workloads,
        const std::vector<DataflowStrategy>& strategies = {
            DataflowStrategy::WEIGHT_STATIONARY,
            DataflowStrategy::INPUT_STATIONARY,
            DataflowStrategy::OUTPUT_STATIONARY
        });

    /**
     * @brief Compute Pareto frontier from evaluations
     */
    ParetoFrontier compute_pareto_frontier(
        const std::vector<ScheduleEvaluation>& evaluations);

    /**
     * @brief Print characterization summary
     */
    void print_summary(const ParetoFrontier& frontier) const;

    /**
     * @brief Export results to CSV for plotting
     */
    void export_csv(
        const std::vector<ScheduleEvaluation>& evaluations,
        const std::string& filename) const;

    /**
     * @brief Export Pareto frontier to CSV
     */
    void export_pareto_csv(
        const ParetoFrontier& frontier,
        const std::string& filename) const;

    // Accessors
    const EnergyModel& energy_model() const { return energy_model_; }
    const LatencyModel& latency_model() const { return latency_model_; }

private:
    TileOptimizer::MemoryHierarchy memory_;
    EnergyModel energy_model_;
    LatencyModel latency_model_;

    TileOptimizer optimizer_;
    L2TileScheduler scheduler_;

    // Helper methods

    /**
     * @brief Calculate ideal performance (lower bound)
     */
    IdealMetrics calculate_ideal(const TensorShape& shape);

    /**
     * @brief Calculate energy from L2 schedule
     */
    double calculate_energy(
        const TensorShape& shape,
        const L2TileScheduler::L2Schedule& schedule);

    /**
     * @brief Calculate latency from L2 schedule
     */
    Cycle calculate_latency(
        const TensorShape& shape,
        const L2TileScheduler::L2Schedule& schedule);

    /**
     * @brief Calculate utilization
     */
    double calculate_utilization(
        const TensorShape& shape,
        const TileOptimizer::TileConfig& config);

    /**
     * @brief Check if point dominates another (for Pareto)
     * Returns true if p1 dominates p2
     */
    bool dominates(const ParetoPoint& p1, const ParetoPoint& p2) const {
        // p1 dominates p2 if p1 is no worse in both dimensions and better in at least one
        bool better_energy = p1.energy <= p2.energy;
        bool better_latency = p1.latency <= p2.latency;
        bool strictly_better = (p1.energy < p2.energy) || (p1.latency < p2.latency);

        return better_energy && better_latency && strictly_better;
    }
};

} // namespace sw::kpu::compiler
