/**
 * @file tile_layout.hpp
 * @brief Memory layout strategies for tensor tiles across memory channels
 *
 * This file defines configurable memory layout policies that determine how
 * tensor tiles (A, B, C for matmul) are mapped to memory channels. The goal
 * is to maximize concurrent bandwidth by ensuring tiles accessed together
 * are on different channels.
 *
 * Layout Policies:
 *   1. MATRIX_PARTITIONED - Each matrix assigned to dedicated channels
 *   2. ROUND_ROBIN - Tiles distributed round-robin across all channels
 *   3. ITERATION_AWARE - A on even channels, B on odd (guarantees no conflicts)
 *   4. HARDWARE_INTERLEAVED - Address bits determine channel (like real HW)
 *
 * Each policy has different trade-offs in terms of:
 *   - Bandwidth utilization
 *   - Memory capacity per tensor
 *   - Implementation complexity
 *   - Debugging ease
 */

#pragma once

#include <sw/kpu/isa/data_movement_isa.hpp>
#include <sw/concepts.hpp>
#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace sw::kpu::isa {

// ============================================================================
// Layout Policy Enumeration
// ============================================================================

/**
 * @brief Memory layout policy selection
 */
enum class LayoutPolicy {
    MATRIX_PARTITIONED,     // Option 1: Matrices assigned to channel subsets
    ROUND_ROBIN,            // Option 2: Tiles round-robin across channels
    ITERATION_AWARE,        // Option 3: A=even channels, B=odd channels
    HARDWARE_INTERLEAVED    // Option 4: Address bits select channel
};

/**
 * @brief Convert policy to string for display
 */
inline std::string layout_policy_to_string(LayoutPolicy policy) {
    switch (policy) {
        case LayoutPolicy::MATRIX_PARTITIONED: return "MATRIX_PARTITIONED";
        case LayoutPolicy::ROUND_ROBIN: return "ROUND_ROBIN";
        case LayoutPolicy::ITERATION_AWARE: return "ITERATION_AWARE";
        case LayoutPolicy::HARDWARE_INTERLEAVED: return "HARDWARE_INTERLEAVED";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Tile Location - Result of layout computation
// ============================================================================

/**
 * @brief Physical location of a tile in the memory system
 */
struct TileLocation {
    uint8_t channel;        // Memory channel (DMA engine)
    Address address;        // Base address within channel's address space
    uint8_t l3_tile_id;     // L3 tile cache slot
    uint8_t l2_bank_id;     // L2 bank assignment

    std::string to_string() const {
        return "Ch" + std::to_string(channel) +
               " @0x" + std::to_string(address) +
               " L3[" + std::to_string(l3_tile_id) + "]" +
               " L2[" + std::to_string(l2_bank_id) + "]";
    }
};

// ============================================================================
// Layout Configuration
// ============================================================================

/**
 * @brief Configuration for tile layout
 */
struct LayoutConfig {
    // Memory system parameters
    uint8_t num_channels = 4;           // Number of memory channels
    uint8_t num_l3_tiles = 4;           // Number of L3 tile slots
    uint8_t num_l2_banks = 8;           // Number of L2 banks
    Size channel_capacity = 256 * 1024 * 1024;  // Bytes per channel (256 MB)

    // Tile parameters
    Size tile_size_bytes = 4096;        // Tile size in bytes
    Size element_size = 4;              // Element size (4 = float32)

    // Matrix dimensions in tiles
    Size m_tiles = 0;                   // Number of tiles in M dimension
    Size n_tiles = 0;                   // Number of tiles in N dimension
    Size k_tiles = 0;                   // Number of tiles in K dimension

    // Hardware interleaving parameters (Option 4)
    Size interleave_granularity = 64;   // Cache line size for HW interleaving

    // Channel assignment for MATRIX_PARTITIONED (Option 1)
    // Specifies which channels each matrix uses
    struct MatrixChannels {
        std::vector<uint8_t> a_channels = {0, 1};  // Channels for A
        std::vector<uint8_t> b_channels = {2};     // Channels for B
        std::vector<uint8_t> c_channels = {3};     // Channels for C
    } matrix_channels;

    // Compute derived values
    Size num_a_tiles() const { return m_tiles * k_tiles; }
    Size num_b_tiles() const { return k_tiles * n_tiles; }
    Size num_c_tiles() const { return m_tiles * n_tiles; }
    Size total_tiles() const { return num_a_tiles() + num_b_tiles() + num_c_tiles(); }
};

// ============================================================================
// Base Class for Tile Layout
// ============================================================================

/**
 * @brief Abstract base class for tile layout strategies
 *
 * Subclasses implement specific layout policies. The interface allows:
 * - Computing the physical location of any tile
 * - Querying which channel a tile is on
 * - Getting the address for DMA operations
 */
class TileLayout {
public:
    virtual ~TileLayout() = default;

    /**
     * @brief Get the layout policy type
     */
    virtual LayoutPolicy policy() const = 0;

    /**
     * @brief Get the physical location of a tile
     * @param matrix Which matrix (A, B, or C)
     * @param ti M-dimension tile index
     * @param tj N-dimension tile index
     * @param tk K-dimension tile index
     * @return Physical location including channel and address
     */
    virtual TileLocation get_tile_location(MatrixID matrix,
                                           Size ti, Size tj, Size tk) const = 0;

    /**
     * @brief Get just the channel for a tile (convenience method)
     */
    uint8_t get_channel(MatrixID matrix, Size ti, Size tj, Size tk) const {
        return get_tile_location(matrix, ti, tj, tk).channel;
    }

    /**
     * @brief Get just the address for a tile (convenience method)
     */
    Address get_address(MatrixID matrix, Size ti, Size tj, Size tk) const {
        return get_tile_location(matrix, ti, tj, tk).address;
    }

    /**
     * @brief Get the configuration
     */
    const LayoutConfig& config() const { return config_; }

    /**
     * @brief Check if two tiles would conflict (same channel)
     */
    bool conflicts(MatrixID mat1, Size ti1, Size tj1, Size tk1,
                   MatrixID mat2, Size ti2, Size tj2, Size tk2) const {
        return get_channel(mat1, ti1, tj1, tk1) == get_channel(mat2, ti2, tj2, tk2);
    }

    /**
     * @brief Generate a report of the layout
     */
    virtual std::string generate_report() const;

protected:
    explicit TileLayout(const LayoutConfig& config) : config_(config) {}
    LayoutConfig config_;
};

// ============================================================================
// Option 1: Matrix-Partitioned Layout
// ============================================================================

/**
 * @brief Each matrix is assigned to a dedicated subset of channels
 *
 * Simple and predictable, but may underutilize some channels if matrices
 * have different access patterns. Easy to debug since channel assignment
 * is purely based on matrix identity.
 *
 * Example with 4 channels:
 *   A -> Channels 0, 1
 *   B -> Channel 2
 *   C -> Channel 3
 */
class MatrixPartitionedLayout : public TileLayout {
public:
    explicit MatrixPartitionedLayout(const LayoutConfig& config);

    LayoutPolicy policy() const override { return LayoutPolicy::MATRIX_PARTITIONED; }

    TileLocation get_tile_location(MatrixID matrix,
                                   Size ti, Size tj, Size tk) const override;

    std::string generate_report() const override;

private:
    // Per-channel tile counters for address calculation
    std::vector<Size> tiles_per_channel_;

    // Get which channels a matrix uses
    const std::vector<uint8_t>& get_matrix_channels(MatrixID matrix) const;

    // Compute linear tile index within a matrix
    Size get_local_tile_index(MatrixID matrix, Size ti, Size tj, Size tk) const;
};

// ============================================================================
// Option 2: Round-Robin Layout
// ============================================================================

/**
 * @brief Tiles distributed round-robin across all channels
 *
 * Even distribution over time, but no guarantee that A and B tiles
 * accessed in the same iteration won't conflict. Simplest implementation.
 *
 * Global tile index = A_offset + local_index (for A)
 *                   = B_offset + local_index (for B)
 *                   = C_offset + local_index (for C)
 * Channel = global_tile_index % num_channels
 */
class RoundRobinLayout : public TileLayout {
public:
    explicit RoundRobinLayout(const LayoutConfig& config);

    LayoutPolicy policy() const override { return LayoutPolicy::ROUND_ROBIN; }

    TileLocation get_tile_location(MatrixID matrix,
                                   Size ti, Size tj, Size tk) const override;

    std::string generate_report() const override;

private:
    Size a_offset_;     // Global offset for A tiles
    Size b_offset_;     // Global offset for B tiles
    Size c_offset_;     // Global offset for C tiles

    Size get_global_tile_index(MatrixID matrix, Size ti, Size tj, Size tk) const;
};

// ============================================================================
// Option 3: Iteration-Aware Layout
// ============================================================================

/**
 * @brief A tiles on even channels, B tiles on odd channels
 *
 * Guarantees that A and B tiles accessed in any iteration are on different
 * channels, eliminating conflicts. C tiles can use any channel since they're
 * accessed at different times (drain phase, not load phase).
 *
 * Channel assignment:
 *   A[ti,tk] -> 2 * ((ti + tk) % (num_channels/2))       = 0, 2, 4, ...
 *   B[tk,tj] -> 2 * ((tk + tj) % (num_channels/2)) + 1   = 1, 3, 5, ...
 *   C[ti,tj] -> 2 * ((ti + tj) % (num_channels/2))       = 0, 2, 4, ...
 */
class IterationAwareLayout : public TileLayout {
public:
    explicit IterationAwareLayout(const LayoutConfig& config);

    LayoutPolicy policy() const override { return LayoutPolicy::ITERATION_AWARE; }

    TileLocation get_tile_location(MatrixID matrix,
                                   Size ti, Size tj, Size tk) const override;

    std::string generate_report() const override;

private:
    Size half_channels_;

    // Track tile placement for address calculation
    // channel -> list of (matrix, ti, tj, tk) placed there
    struct TilePlacement {
        MatrixID matrix;
        Size ti, tj, tk;
        Address address;
    };
    std::vector<std::vector<TilePlacement>> channel_placements_;

    void precompute_placements();
    Address lookup_address(MatrixID matrix, Size ti, Size tj, Size tk) const;
};

// ============================================================================
// Option 4: Hardware-Interleaved Layout
// ============================================================================

/**
 * @brief Address bits determine channel, like real hardware
 *
 * Channel = (address / interleave_granularity) % num_channels
 *
 * Tiles are placed at addresses that put them on desired channels.
 * Most realistic but also most complex. Tile sizes may need to be
 * multiples of interleave_granularity * num_channels for clean mapping.
 */
class HardwareInterleavedLayout : public TileLayout {
public:
    explicit HardwareInterleavedLayout(const LayoutConfig& config);

    LayoutPolicy policy() const override { return LayoutPolicy::HARDWARE_INTERLEAVED; }

    TileLocation get_tile_location(MatrixID matrix,
                                   Size ti, Size tj, Size tk) const override;

    std::string generate_report() const override;

private:
    // Precomputed addresses for all tiles
    std::vector<Address> a_addresses_;
    std::vector<Address> b_addresses_;
    std::vector<Address> c_addresses_;

    // Convert address to channel
    uint8_t address_to_channel(Address addr) const;

    // Find next address on target channel
    Address find_address_on_channel(Address min_addr, uint8_t target_channel) const;

    // Precompute all tile addresses
    void precompute_addresses();

    // Get local tile index
    Size get_local_tile_index(MatrixID matrix, Size ti, Size tj, Size tk) const;
};

// ============================================================================
// Factory Function
// ============================================================================

/**
 * @brief Create a tile layout with the specified policy
 */
std::unique_ptr<TileLayout> create_tile_layout(LayoutPolicy policy,
                                                const LayoutConfig& config);

} // namespace sw::kpu::isa
