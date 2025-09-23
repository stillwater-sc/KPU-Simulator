#include <vector>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cstring>

#include <sw/kpu/components/block_mover.hpp>
#include <sw/kpu/components/l3_tile.hpp>
#include <sw/kpu/components/l2_bank.hpp>

namespace sw::kpu {

// BlockMover implementation - manages L3↔L2 data movement with transformations
BlockMover::BlockMover(size_t engine_id, size_t associated_l3_tile_id)
    : is_active(false), engine_id(engine_id), associated_l3_tile_id(associated_l3_tile_id) {
}

void BlockMover::enqueue_block_transfer(size_t src_l3_tile_id, Address src_offset,
                                       size_t dst_l2_bank_id, Address dst_offset,
                                       Size block_height, Size block_width, Size element_size,
                                       TransformType transform,
                                       std::function<void()> callback) {
    transfer_queue.emplace_back(BlockTransfer{
        src_l3_tile_id, src_offset,
        dst_l2_bank_id, dst_offset,
        block_height, block_width, element_size,
        transform, std::move(callback)
    });
}

bool BlockMover::process_transfers(std::vector<L3Tile>& l3_tiles,
                                  std::vector<L2Bank>& l2_banks) {
    if (transfer_queue.empty()) {
        is_active = false;
        return false;
    }

    is_active = true;
    auto& transfer = transfer_queue.front();

    // Validate indices
    if (transfer.src_l3_tile_id >= l3_tiles.size()) {
        throw std::out_of_range("Invalid L3 tile ID: " + std::to_string(transfer.src_l3_tile_id));
    }
    if (transfer.dst_l2_bank_id >= l2_banks.size()) {
        throw std::out_of_range("Invalid L2 bank ID: " + std::to_string(transfer.dst_l2_bank_id));
    }

    // Calculate total block size
    Size block_size = transfer.block_height * transfer.block_width * transfer.element_size;

    // Allocate buffers for the transfer
    std::vector<std::uint8_t> src_buffer(block_size);
    std::vector<std::uint8_t> dst_buffer(block_size);

    // Read block from L3 tile
    l3_tiles[transfer.src_l3_tile_id].read_block(
        transfer.src_offset, src_buffer.data(),
        transfer.block_height, transfer.block_width, transfer.element_size
    );

    // Apply transformation
    apply_transform(src_buffer, dst_buffer, transfer);

    // Write transformed block to L2 bank
    l2_banks[transfer.dst_l2_bank_id].write_block(
        transfer.dst_offset, dst_buffer.data(),
        transfer.block_height, transfer.block_width, transfer.element_size
    );

    // Call completion callback if provided
    if (transfer.completion_callback) {
        transfer.completion_callback();
    }

    transfer_queue.erase(transfer_queue.begin());

    bool completed = transfer_queue.empty();
    if (completed) {
        is_active = false;
    }

    return completed;
}

void BlockMover::apply_transform(const std::vector<uint8_t>& src_data,
                                std::vector<uint8_t>& dst_data,
                                const BlockTransfer& transfer) {
    switch (transfer.transform) {
        case TransformType::IDENTITY:
            identity_copy(src_data, dst_data);
            break;

        case TransformType::TRANSPOSE:
            transpose_block(src_data, dst_data,
                          transfer.block_height, transfer.block_width, transfer.element_size);
            break;

        case TransformType::BLOCK_RESHAPE:
        case TransformType::SHUFFLE_PATTERN:
            // For now, fall back to identity copy
            // These will be implemented in future iterations
            identity_copy(src_data, dst_data);
            break;

        default:
            throw std::runtime_error("Unknown transform type");
    }
}

void BlockMover::identity_copy(const std::vector<uint8_t>& src,
                              std::vector<uint8_t>& dst) {
    std::copy(src.begin(), src.end(), dst.begin());
}

void BlockMover::transpose_block(const std::vector<uint8_t>& src,
                                std::vector<uint8_t>& dst,
                                Size height, Size width, Size element_size) {
    // Transpose a 2D block: (i,j) -> (j,i)
    for (Size row = 0; row < height; ++row) {
        for (Size col = 0; col < width; ++col) {
            Size src_offset = (row * width + col) * element_size;
            Size dst_offset = (col * height + row) * element_size;

            // Copy element from (row,col) to (col,row)
            std::memcpy(dst.data() + dst_offset,
                       src.data() + src_offset,
                       element_size);
        }
    }
}

void BlockMover::reset() {
    transfer_queue.clear();
    is_active = false;
}

} // namespace sw::kpu