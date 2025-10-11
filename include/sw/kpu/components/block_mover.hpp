#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <functional>

// Windows/MSVC compatibility
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251) // DLL interface warnings
    #ifdef BUILDING_KPU_SIMULATOR
        #define KPU_API __declspec(dllexport)
    #else
        #define KPU_API __declspec(dllimport)
    #endif
#else
    #define KPU_API
#endif

#include <sw/concepts.hpp>

namespace sw::kpu {

// Forward declarations for L3 and L2 memory components
class L3Tile;
class L2Bank;

// Block Mover for data movement between L3 tiles and L2 cache banks
class KPU_API BlockMover {
public:
    enum class TransformType {
        IDENTITY,           // Direct copy (no transformation)
        TRANSPOSE,          // Matrix transpose (row↔column)
        BLOCK_RESHAPE,      // Reshape for different tiling patterns
        SHUFFLE_PATTERN     // Custom shuffle network operation
    };

    struct BlockTransfer {
        // Source (L3 tile)
        size_t src_l3_tile_id;
        Address src_offset;

        // Destination (L2 bank)
        size_t dst_l2_bank_id;
        Address dst_offset;

        // Transfer geometry (2D block dimensions)
        Size block_height;
        Size block_width;
        Size element_size;

        // Transformation type
        TransformType transform;

        // Completion callback
        std::function<void()> completion_callback;
    };

private:
    std::vector<BlockTransfer> transfer_queue;
    bool is_active;
    size_t engine_id;              // For debugging/identification
    size_t associated_l3_tile_id;  // Which L3 tile this block mover serves

    // Multi-cycle timing state
    Cycle cycles_remaining;        // Cycles left for current transfer
    std::vector<uint8_t> transfer_buffer;  // Buffer for current transfer data

    // Internal transformation engine
    void apply_transform(const std::vector<uint8_t>& src_data,
                        std::vector<uint8_t>& dst_data,
                        const BlockTransfer& transfer);

    // Specific transformation implementations
    void identity_copy(const std::vector<uint8_t>& src,
                      std::vector<uint8_t>& dst);

    void transpose_block(const std::vector<uint8_t>& src,
                        std::vector<uint8_t>& dst,
                        Size height, Size width, Size element_size);

public:
    explicit BlockMover(size_t engine_id, size_t associated_l3_tile_id);
    ~BlockMover() = default;

    // Block transfer operations - configured per-transfer
    void enqueue_block_transfer(size_t src_l3_tile_id, Address src_offset,
                               size_t dst_l2_bank_id, Address dst_offset,
                               Size block_height, Size block_width, Size element_size,
                               TransformType transform = TransformType::IDENTITY,
                               std::function<void()> callback = nullptr);

    // Process queued transfers
    bool process_transfers(std::vector<L3Tile>& l3_tiles,
                          std::vector<L2Bank>& l2_banks);

    // Status and control
    bool is_busy() const { return is_active || !transfer_queue.empty(); }
    void reset();

    // Configuration queries
    size_t get_engine_id() const { return engine_id; }
    size_t get_associated_l3_tile() const { return associated_l3_tile_id; }
    size_t get_queue_size() const { return transfer_queue.size(); }
};

} // namespace sw::kpu

#ifdef _MSC_VER
    #pragma warning(pop)
#endif