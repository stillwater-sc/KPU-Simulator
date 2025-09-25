#include <sw/kpu/components/buffet_integration.hpp>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cstring>

namespace sw::kpu {

// BuffetBlockMoverAdapter Implementation
BuffetBlockMoverAdapter::BuffetBlockMoverAdapter(Buffet* buffet, size_t adapter_id)
    : buffet(buffet), adapter_id(adapter_id) {
    if (buffet) {
        // Register this adapter - actual BlockMover registration would happen here
    }
}

void BuffetBlockMoverAdapter::orchestrated_block_transfer(
    size_t src_l3_tile_id, Address src_offset,
    size_t dst_buffet_bank, Address dst_offset,
    Size block_height, Size block_width, Size element_size,
    BlockMover::TransformType transform,
    std::function<void()> completion_callback) {

    if (!buffet) return;

    Size transfer_size = block_height * block_width * element_size;

    // Create EDDO workflow for the block transfer
    EDDOWorkflowBuilder builder;
    builder.prefetch(dst_buffet_bank, src_offset, dst_offset, transfer_size);

    if (completion_callback) {
        builder.compute(dst_buffet_bank, completion_callback);
    }

    builder.execute_on(*buffet);
}

void BuffetBlockMoverAdapter::double_buffered_transfer(
    size_t src_l3_tile_id, Address src_offset,
    size_t primary_bank, size_t secondary_bank,
    Size transfer_size) {

    if (!buffet) return;

    buffet->orchestrate_double_buffer(primary_bank, secondary_bank, src_offset, transfer_size);
}

void BuffetBlockMoverAdapter::pipelined_transfer_compute(
    size_t src_l3_tile_id, Address src_offset,
    size_t input_bank, size_t output_bank,
    Size transfer_size,
    const std::function<void()>& compute_operation) {

    if (!buffet) return;

    buffet->orchestrate_pipeline_stage(input_bank, output_bank, compute_operation);
}

bool BuffetBlockMoverAdapter::is_busy() const {
    return buffet ? buffet->is_busy() : false;
}

void BuffetBlockMoverAdapter::reset() {
    if (buffet) {
        buffet->reset();
    }
}

// BuffetStreamerAdapter Implementation
BuffetStreamerAdapter::BuffetStreamerAdapter(Buffet* buffet, size_t adapter_id)
    : buffet(buffet), adapter_id(adapter_id) {
    if (buffet) {
        // Register this adapter - actual Streamer registration would happen here
    }
}

void BuffetStreamerAdapter::start_eddo_stream(const EDDOStreamConfig& config) {
    if (!buffet) return;

    Size stream_size = config.matrix_height * config.matrix_width * config.element_size;

    // Create streaming EDDO workflow
    EDDOWorkflowBuilder builder;

    if (config.enable_prefetch_pipelining) {
        // Pipeline multiple prefetch stages
        for (size_t depth = 0; depth < config.prefetch_depth; ++depth) {
            Address prefetch_addr = config.buffet_base_addr + (depth * stream_size / config.prefetch_depth);
            Size prefetch_size = stream_size / config.prefetch_depth;

            builder.prefetch(config.buffet_bank_id, prefetch_addr, 0, prefetch_size);
        }
    } else {
        // Single prefetch stage
        builder.prefetch(config.buffet_bank_id, config.buffet_base_addr, 0, stream_size);
    }

    // Add compute phase for streaming
    builder.compute(config.buffet_bank_id, [config]() {
        if (config.completion_callback) {
            config.completion_callback();
        }
    });

    // Execute based on direction
    if (config.direction == Streamer::StreamDirection::L1_TO_L2) {
        builder.writeback(config.buffet_bank_id, 0, config.l1_base_addr, stream_size);
    }

    builder.execute_on(*buffet);
}

void BuffetStreamerAdapter::orchestrated_systolic_stream(
    size_t input_bank, size_t weight_bank, size_t output_bank,
    size_t l1_scratchpad, Size matrix_dim) {

    if (!buffet) return;

    Size matrix_size = matrix_dim * matrix_dim * sizeof(float);

    // Create systolic array streaming pattern
    EDDOWorkflowBuilder builder;

    // Prefetch input matrix
    builder.prefetch(input_bank, 0, 0, matrix_size);

    // Prefetch weight matrix
    builder.prefetch(weight_bank, 0, 0, matrix_size);

    // Compute phase (systolic array operation)
    builder.compute(output_bank, [=]() {
        // Simulate systolic array computation timing
        std::this_thread::sleep_for(std::chrono::microseconds(matrix_dim));
    });

    // Stream results back
    builder.writeback(output_bank, 0, 0, matrix_size);

    builder.execute_on(*buffet);
}

void BuffetStreamerAdapter::multi_phase_stream(const std::vector<EDDOStreamConfig>& phases) {
    for (const auto& phase : phases) {
        start_eddo_stream(phase);
    }
}

bool BuffetStreamerAdapter::is_streaming() const {
    return buffet ? buffet->is_busy() : false;
}

void BuffetStreamerAdapter::abort_stream() {
    if (buffet) {
        buffet->abort_pending_commands();
    }
}

void BuffetStreamerAdapter::reset() {
    if (buffet) {
        buffet->reset();
    }
}

// EDDOMatrixOrchestrator Implementation
EDDOMatrixOrchestrator::EDDOMatrixOrchestrator(Buffet* buffet, size_t orchestrator_id)
    : buffet(buffet), orchestrator_id(orchestrator_id) {
}

void EDDOMatrixOrchestrator::register_block_mover(BlockMover* mover) {
    if (buffet && mover) {
        block_adapters.emplace_back(buffet, block_adapters.size());
        buffet->register_block_mover(mover);
    }
}

void EDDOMatrixOrchestrator::register_streamer(Streamer* streamer) {
    if (buffet && streamer) {
        stream_adapters.emplace_back(buffet, stream_adapters.size());
        buffet->register_streamer(streamer);
    }
}

void EDDOMatrixOrchestrator::orchestrate_matrix_multiply(const MatrixOperationConfig& config) {
    if (!buffet) return;

    // Calculate tiling parameters
    size_t num_tiles_m = (config.m + config.tile_size_m - 1) / config.tile_size_m;
    size_t num_tiles_n = (config.n + config.tile_size_n - 1) / config.tile_size_n;
    size_t num_tiles_k = (config.k + config.tile_size_k - 1) / config.tile_size_k;

    // Create comprehensive EDDO workflow for tiled matrix multiplication
    EDDOWorkflowBuilder workflow;

    size_t sequence_counter = 1;

    for (size_t tile_i = 0; tile_i < num_tiles_m; ++tile_i) {
        for (size_t tile_j = 0; tile_j < num_tiles_n; ++tile_j) {
            // For each output tile C[i,j], we need to compute sum over k
            for (size_t tile_k = 0; tile_k < num_tiles_k; ++tile_k) {

                // Calculate actual tile dimensions (handle edge cases)
                Size actual_m = std::min(config.tile_size_m, config.m - tile_i * config.tile_size_m);
                Size actual_n = std::min(config.tile_size_n, config.n - tile_j * config.tile_size_n);
                Size actual_k = std::min(config.tile_size_k, config.k - tile_k * config.tile_size_k);

                Size tile_a_size = actual_m * actual_k * config.element_size;
                Size tile_b_size = actual_k * actual_n * config.element_size;
                Size tile_c_size = actual_m * actual_n * config.element_size;

                // Determine bank allocation (round-robin)
                size_t bank_a = config.input_banks[tile_k % config.input_banks.size()];
                size_t bank_b = config.input_banks[(tile_k + 1) % config.input_banks.size()];
                size_t bank_c = config.output_banks[(tile_i * num_tiles_n + tile_j) % config.output_banks.size()];

                // Calculate memory addresses for this tile
                Address addr_a = config.matrix_a_addr +
                    (tile_i * config.tile_size_m * config.k + tile_k * config.tile_size_k) * config.element_size;
                Address addr_b = config.matrix_b_addr +
                    (tile_k * config.tile_size_k * config.n + tile_j * config.tile_size_n) * config.element_size;
                Address addr_c = config.matrix_c_addr +
                    (tile_i * config.tile_size_m * config.n + tile_j * config.tile_size_n) * config.element_size;

                // Prefetch A tile
                workflow.prefetch(bank_a, addr_a, 0, tile_a_size);

                // Prefetch B tile
                workflow.prefetch(bank_b, addr_b, 0, tile_b_size);

                // Compute phase
                workflow.compute(bank_c, [=, &config, this]() {
                    if (config.compute_kernel) {
                        // Load tiles from buffet banks
                        std::vector<float> tile_a_data(actual_m * actual_k);
                        std::vector<float> tile_b_data(actual_k * actual_n);
                        std::vector<float> tile_c_data(actual_m * actual_n, 0.0f);

                        buffet->read(bank_a, 0, tile_a_data.data(), tile_a_size);
                        buffet->read(bank_b, 0, tile_b_data.data(), tile_b_size);

                        // Execute compute kernel
                        config.compute_kernel(tile_a_data, tile_b_data, tile_c_data,
                                            actual_m, actual_n, actual_k);

                        // Store result back to buffet
                        buffet->write(bank_c, 0, tile_c_data.data(), tile_c_size);
                    }
                });

                // Writeback C tile
                workflow.writeback(bank_c, 0, addr_c, tile_c_size);
            }
        }
    }

    // Add final synchronization
    workflow.sync();

    // Execute the complete workflow
    workflow.execute_on(*buffet);
}

void EDDOMatrixOrchestrator::orchestrate_matrix_transpose(
    Address src_addr, Address dst_addr, Size rows, Size cols, size_t element_size) {

    if (!buffet) return;

    constexpr size_t TILE_SIZE = 64; // 64x64 tiles for cache efficiency
    Size matrix_size = rows * cols * element_size;

    EDDOWorkflowBuilder workflow;

    // Process in tiles for better cache behavior
    for (size_t tile_row = 0; tile_row < rows; tile_row += TILE_SIZE) {
        for (size_t tile_col = 0; tile_col < cols; tile_col += TILE_SIZE) {

            Size actual_rows = std::min(TILE_SIZE, rows - tile_row);
            Size actual_cols = std::min(TILE_SIZE, cols - tile_col);
            Size tile_size = actual_rows * actual_cols * element_size;

            // Use different banks for input and output tiles
            size_t input_bank = (tile_row / TILE_SIZE) % buffet->get_num_banks();
            size_t output_bank = (input_bank + 1) % buffet->get_num_banks();

            Address tile_src_addr = src_addr + (tile_row * cols + tile_col) * element_size;
            Address tile_dst_addr = dst_addr + (tile_col * rows + tile_row) * element_size;

            workflow.prefetch(input_bank, tile_src_addr, 0, tile_size);

            workflow.compute(output_bank, [=, this]() {
                // Transpose tile in-memory
                std::vector<uint8_t> src_tile(tile_size);
                std::vector<uint8_t> dst_tile(tile_size);

                buffet->read(input_bank, 0, src_tile.data(), tile_size);

                // Perform transpose on the tile
                for (size_t r = 0; r < actual_rows; ++r) {
                    for (size_t c = 0; c < actual_cols; ++c) {
                        size_t src_offset = (r * actual_cols + c) * element_size;
                        size_t dst_offset = (c * actual_rows + r) * element_size;
                        std::memcpy(&dst_tile[dst_offset], &src_tile[src_offset], element_size);
                    }
                }

                buffet->write(output_bank, 0, dst_tile.data(), tile_size);
            });

            workflow.writeback(output_bank, 0, tile_dst_addr, tile_size);
        }
    }

    workflow.sync();
    workflow.execute_on(*buffet);
}

void EDDOMatrixOrchestrator::orchestrate_convolution(
    Address input_addr, Address kernel_addr, Address output_addr,
    Size input_h, Size input_w, Size kernel_h, Size kernel_w, Size channels) {

    if (!buffet) return;

    Size output_h = input_h - kernel_h + 1;
    Size output_w = input_w - kernel_w + 1;

    EDDOWorkflowBuilder workflow;

    // Simplified convolution implementation
    // In practice, this would use more sophisticated tiling and optimization
    for (size_t ch = 0; ch < channels; ++ch) {
        size_t input_bank = ch % buffet->get_num_banks();
        size_t kernel_bank = (ch + 1) % buffet->get_num_banks();
        size_t output_bank = (ch + 2) % buffet->get_num_banks();

        Size input_channel_size = input_h * input_w * sizeof(float);
        Size kernel_channel_size = kernel_h * kernel_w * sizeof(float);
        Size output_channel_size = output_h * output_w * sizeof(float);

        workflow.prefetch(input_bank, input_addr + ch * input_channel_size, 0, input_channel_size);
        workflow.prefetch(kernel_bank, kernel_addr + ch * kernel_channel_size, 0, kernel_channel_size);

        workflow.compute(output_bank, [=, this]() {
            // Convolution computation would go here
            // For now, just allocate output space
            std::vector<float> output_data(output_h * output_w, 0.0f);
            buffet->write(output_bank, 0, output_data.data(), output_channel_size);
        });

        workflow.writeback(output_bank, 0, output_addr + ch * output_channel_size, output_channel_size);
    }

    workflow.sync();
    workflow.execute_on(*buffet);
}

bool EDDOMatrixOrchestrator::is_busy() const {
    if (!buffet) return false;

    if (buffet->is_busy()) return true;

    for (const auto& adapter : block_adapters) {
        if (adapter.is_busy()) return true;
    }

    for (const auto& adapter : stream_adapters) {
        if (adapter.is_streaming()) return true;
    }

    return false;
}

void EDDOMatrixOrchestrator::wait_for_completion() {
    while (is_busy()) {
        if (buffet) {
            buffet->process_eddo_commands();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void EDDOMatrixOrchestrator::abort_all_operations() {
    if (buffet) {
        buffet->abort_pending_commands();
    }

    for (auto& adapter : stream_adapters) {
        adapter.abort_stream();
    }
}

void EDDOMatrixOrchestrator::reset() {
    if (buffet) {
        buffet->reset();
    }

    for (auto& adapter : block_adapters) {
        adapter.reset();
    }

    for (auto& adapter : stream_adapters) {
        adapter.reset();
    }
}

EDDOMatrixOrchestrator::OrchestrationMetrics EDDOMatrixOrchestrator::get_metrics() const {
    OrchestrationMetrics metrics{};

    if (buffet) {
        auto buffet_metrics = buffet->get_performance_metrics();
        metrics.total_bytes_moved = buffet_metrics.total_read_accesses + buffet_metrics.total_write_accesses;
        metrics.average_bank_utilization = buffet_metrics.average_bank_utilization;
        metrics.total_operations_completed = buffet_metrics.completed_eddo_commands;
        metrics.operation_efficiency = buffet_metrics.total_cache_hits /
            static_cast<double>(buffet_metrics.total_cache_hits + buffet_metrics.total_cache_misses + 1);
    }

    return metrics;
}

} // namespace sw::kpu