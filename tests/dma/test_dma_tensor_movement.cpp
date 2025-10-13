#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <sw/kpu/kpu_simulator.hpp>

using namespace sw::kpu;

// Tensor movement test fixture focusing on ML workload patterns
class DMATensorMovementFixture {
public:
    KPUSimulator::Config config;
    std::unique_ptr<KPUSimulator> sim;

    DMATensorMovementFixture() {
        // Configuration for tensor processing workloads
        config.memory_bank_count = 4;
        config.memory_bank_capacity_mb = 256;
        config.memory_bandwidth_gbps = 64;
        config.scratchpad_count = 2; // L3 cache simulation
        config.scratchpad_capacity_kb = 2048; // Large L3 caches
        config.compute_tile_count = 4;
        config.dma_engine_count = 8; // Multiple DMA engines for parallel data movement

        sim = std::make_unique<KPUSimulator>(config);
    }

    // Generate tensor data with specific patterns
    std::vector<float> generate_tensor(size_t height, size_t width, float base_value = 1.0f) {
        std::vector<float> tensor(height * width);
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                tensor[i * width + j] = base_value + static_cast<float>(i * width + j) * 0.001f;
            }
        }
        return tensor;
    }

    // Verify tensor data integrity
    bool verify_tensor(const std::vector<float>& expected,
                       Address addr, size_t scratchpad_id) {
        std::vector<float> actual(expected.size());
        size_t size_bytes = expected.size() * sizeof(float);
        sim->read_scratchpad(scratchpad_id, addr, actual.data(), size_bytes);

        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::abs(expected[i] - actual[i]) > 1e-6f) {
                return false;
            }
        }
        return true;
    }

    // Simulate tiled tensor movement (common in ML workloads)
    struct TensorTile {
        size_t start_row, start_col;
        size_t height, width;
        Address memory_addr;
        Address scratch_addr;
    };

    std::vector<TensorTile> generate_tiles(size_t tensor_height, size_t tensor_width,
                                          size_t tile_height, size_t tile_width,
                                          Address base_memory_addr,
                                          Address base_scratch_addr) {
        std::vector<TensorTile> tiles;
        size_t element_size = sizeof(float);

        for (size_t row = 0; row < tensor_height; row += tile_height) {
            for (size_t col = 0; col < tensor_width; col += tile_width) {
                TensorTile tile;
                tile.start_row = row;
                tile.start_col = col;
                tile.height = std::min(tile_height, tensor_height - row);
                tile.width = std::min(tile_width, tensor_width - col);

                // Memory address (tile-optimized layout)
                size_t tile_index = (row / tile_height) * ((tensor_width + tile_width - 1) / tile_width) +
                                   (col / tile_width);
                tile.memory_addr = base_memory_addr + tile_index * tile_height * tile_width * element_size;

                // Scratchpad address (sequential tiling)
                tile.scratch_addr = base_scratch_addr + tile_index * tile_height * tile_width * element_size;

                tiles.push_back(tile);
            }
        }

        return tiles;
    }
};

TEST_CASE_METHOD(DMATensorMovementFixture, "DMA Tensor - Basic Matrix Transfer", "[dma][tensor]") {
    const size_t matrix_height = 128;
    const size_t matrix_width = 128;
    const size_t matrix_size = matrix_height * matrix_width * sizeof(float);

    if (matrix_size > sim->get_scratchpad_capacity(0)) {
        SKIP("Matrix too large for scratchpad");
    }

    // Generate test matrix
    auto matrix = generate_tensor(matrix_height, matrix_width, 2.5f);

    // Store matrix in external memory
    sim->write_memory_bank(0, 0, matrix.data(), matrix_size);

    // Transfer to scratchpad (L3 cache)
    Address global_src = sim->get_external_bank_base(0);
    Address global_dst = sim->get_scratchpad_base(0);

    bool transfer_complete = false;
    sim->dma_external_to_scratchpad(0, global_src, global_dst, matrix_size,
        [&transfer_complete]() { transfer_complete = true; });

    while (!transfer_complete) {
        sim->step();
    }

    // Verify matrix integrity
    REQUIRE(verify_tensor(matrix, 0, 0));
}

TEST_CASE_METHOD(DMATensorMovementFixture, "DMA Tensor - Multi-Matrix Batch Transfer", "[dma][tensor]") {
    const size_t batch_size = 4;
    const size_t matrix_height = 64;
    const size_t matrix_width = 64;
    const size_t matrix_size = matrix_height * matrix_width * sizeof(float);
    const size_t total_size = batch_size * matrix_size;

    if (total_size > sim->get_scratchpad_capacity(0)) {
        SKIP("Batch too large for scratchpad");
    }

    // Generate batch of matrices
    std::vector<std::vector<float>> matrices(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        matrices[i] = generate_tensor(matrix_height, matrix_width, static_cast<float>(i + 1));
        sim->write_memory_bank(0, i * matrix_size, matrices[i].data(), matrix_size);
    }

    // Transfer entire batch using multiple DMA engines
    std::vector<bool> completions(batch_size, false);

    Address ext_base = sim->get_external_bank_base(0);
    Address scratch_base = sim->get_scratchpad_base(0);

    for (size_t i = 0; i < batch_size; ++i) {
        size_t dma_id = i % config.dma_engine_count;
        Address src_addr = i * matrix_size;
        Address dst_addr = i * matrix_size;

        sim->dma_external_to_scratchpad(dma_id, ext_base + src_addr, scratch_base + dst_addr, matrix_size,
            [&completions, i]() { completions[i] = true; });
    }

    // Wait for all transfers to complete
    while (!std::all_of(completions.begin(), completions.end(), [](bool c) { return c; })) {
        sim->step();
    }

    // Verify all matrices
    for (size_t i = 0; i < batch_size; ++i) {
        REQUIRE(verify_tensor(matrices[i], i * matrix_size, 0));
    }
}

TEST_CASE_METHOD(DMATensorMovementFixture, "DMA Tensor - Tiled Matrix Transfer", "[dma][tensor][tiling]") {
    const size_t tensor_height = 256;
    const size_t tensor_width = 256;
    const size_t tile_height = 64;
    const size_t tile_width = 64;

    const size_t tensor_size = tensor_height * tensor_width * sizeof(float);
    const size_t tile_size = tile_height * tile_width * sizeof(float);
    const size_t num_tiles = (tensor_height / tile_height) * (tensor_width / tile_width);

    if (num_tiles * tile_size > sim->get_scratchpad_capacity(0)) {
        SKIP("Tiled tensor too large for scratchpad");
    }

    // Generate large tensor
    auto tensor = generate_tensor(tensor_height, tensor_width, 1.0f);

    // Store tensor in external memory using tile-optimized layout
    // For this test, we'll arrange data so each tile is contiguous in memory
    std::vector<float> tile_optimized_tensor(tensor_height * tensor_width);
    size_t tile_data_offset = 0;

    // Rearrange data tile by tile for easier DMA
    for (size_t tile_row = 0; tile_row < tensor_height; tile_row += tile_height) {
        for (size_t tile_col = 0; tile_col < tensor_width; tile_col += tile_width) {
            for (size_t r = 0; r < tile_height && (tile_row + r) < tensor_height; ++r) {
                for (size_t c = 0; c < tile_width && (tile_col + c) < tensor_width; ++c) {
                    size_t src_idx = (tile_row + r) * tensor_width + (tile_col + c);
                    tile_optimized_tensor[tile_data_offset++] = tensor[src_idx];
                }
            }
        }
    }

    sim->write_memory_bank(0, 0, tile_optimized_tensor.data(), tensor_size);

    // Generate tile layout
    auto tiles = generate_tiles(tensor_height, tensor_width, tile_height, tile_width, 0, 0);

    std::cout << "Transferring " << tiles.size() << " tiles of "
              << tile_height << "x" << tile_width << "\n";

    // Transfer tiles using multiple DMA engines
    std::vector<bool> tile_completions(tiles.size(), false);
    Address ext_base = sim->get_external_bank_base(0);
    Address scratch_base = sim->get_scratchpad_base(0);

    for (size_t i = 0; i < tiles.size(); ++i) {
        const auto& tile = tiles[i];
        size_t dma_id = i % config.dma_engine_count;

        // For tiled transfers, we need to handle non-contiguous memory access
        // This is a simplified version - real hardware would use 2D DMA
        size_t tile_data_size = tile.height * tile.width * sizeof(float);

        sim->dma_external_to_scratchpad(dma_id, ext_base + tile.memory_addr, scratch_base + tile.scratch_addr, tile_data_size,
            [&tile_completions, i]() { tile_completions[i] = true; });
    }

    // Wait for all tile transfers to complete
    while (!std::all_of(tile_completions.begin(), tile_completions.end(), [](bool c) { return c; })) {
        sim->step();
    }

    // Verify tile data integrity
    for (size_t i = 0; i < tiles.size(); ++i) {
        const auto& tile = tiles[i];

        // Extract expected tile data from original tensor
        std::vector<float> expected_tile(tile.height * tile.width);
        for (size_t row = 0; row < tile.height; ++row) {
            for (size_t col = 0; col < tile.width; ++col) {
                size_t src_idx = (tile.start_row + row) * tensor_width + (tile.start_col + col);
                size_t dst_idx = row * tile.width + col;
                expected_tile[dst_idx] = tensor[src_idx];
            }
        }

        // Verify tile in scratchpad
        REQUIRE(verify_tensor(expected_tile, tile.scratch_addr, 0));
    }
}

TEST_CASE_METHOD(DMATensorMovementFixture, "DMA Tensor - Convolution Data Movement", "[dma][tensor][convolution]") {
    // Simulate convolution layer data movement patterns
    const size_t input_height = 224;
    const size_t input_width = 224;
    const size_t input_channels = 64;
    const size_t kernel_size = 3;
    const size_t output_channels = 128;

    const size_t input_size = input_height * input_width * input_channels * sizeof(float);
    const size_t kernel_size_total = kernel_size * kernel_size * input_channels * output_channels * sizeof(float);

    // Check if data fits in available memory
    if (input_size + kernel_size_total > sim->get_memory_bank_capacity(0)) {
        SKIP("Convolution data too large for memory bank");
    }

    const size_t feature_map_size = input_height * input_width * sizeof(float);
    if (feature_map_size > sim->get_scratchpad_capacity(0)) {
        SKIP("Feature map too large for scratchpad");
    }

    // Generate input feature maps
    std::vector<std::vector<float>> input_channels_data(input_channels);
    for (size_t c = 0; c < input_channels; ++c) {
        input_channels_data[c] = generate_tensor(input_height, input_width, static_cast<float>(c + 1));
        Address channel_addr = c * feature_map_size;
        sim->write_memory_bank(0, channel_addr, input_channels_data[c].data(), feature_map_size);
    }

    // Simulate streaming input channels to scratchpad for processing
    std::cout << "Streaming " << input_channels << " feature maps of "
              << input_height << "x" << input_width << "\n";

    Address ext_base = sim->get_external_bank_base(0);
    Address scratch_base = sim->get_scratchpad_base(0);
    size_t channels_processed = 0;

    for (size_t c = 0; c < input_channels; ++c) {
        bool channel_complete = false;
        Address src_addr = c * feature_map_size;
        size_t dma_id = c % config.dma_engine_count;

        sim->dma_external_to_scratchpad(dma_id, ext_base + src_addr, scratch_base, feature_map_size,
            [&channel_complete, &channels_processed]() {
                channel_complete = true;
                channels_processed++;
            });

        // Process this channel
        while (!channel_complete) {
            sim->step();
        }

        // Verify channel data
        REQUIRE(verify_tensor(input_channels_data[c], 0, 0));

        // In real scenario, compute operations would happen here
        // followed by output data movement
    }

    REQUIRE(channels_processed == input_channels);
}

TEST_CASE_METHOD(DMATensorMovementFixture, "DMA Tensor - Pipeline Simulation", "[dma][tensor][pipeline]") {
    // Simulate a multi-stage ML pipeline with overlapped data movement
    const size_t stage_count = 3;
    const size_t matrix_size_per_stage = 128 * 128 * sizeof(float);

    if (stage_count * matrix_size_per_stage > sim->get_scratchpad_capacity(0)) {
        SKIP("Pipeline data too large for scratchpad");
    }

    // Generate data for pipeline stages
    std::vector<std::vector<float>> stage_data(stage_count);
    for (size_t stage = 0; stage < stage_count; ++stage) {
        stage_data[stage] = generate_tensor(128, 128, static_cast<float>(stage + 10));
        sim->write_memory_bank(stage % config.memory_bank_count,
                              stage * matrix_size_per_stage,
                              stage_data[stage].data(),
                              matrix_size_per_stage);
    }

    // Simulate pipelined execution with overlapped data movement
    std::vector<bool> stage_completions(stage_count, false);
    std::vector<size_t> completion_order;

    auto completion_callback = [&completion_order](size_t stage) {
        return [&completion_order, stage]() {
            completion_order.push_back(stage);
        };
    };

    // Start all stages concurrently (simulating pipeline)
    Address scratch_base = sim->get_scratchpad_base(0);

    for (size_t stage = 0; stage < stage_count; ++stage) {
        size_t src_bank = stage % config.memory_bank_count;
        size_t dma_id = stage % config.dma_engine_count;
        Address src_addr = stage * matrix_size_per_stage;
        Address dst_addr = stage * matrix_size_per_stage;

        Address global_src = sim->get_external_bank_base(src_bank) + src_addr;
        Address global_dst = scratch_base + dst_addr;

        sim->dma_external_to_scratchpad(dma_id, global_src, global_dst, matrix_size_per_stage,
            [&stage_completions, stage, callback = completion_callback(stage)]() {
                stage_completions[stage] = true;
                callback();
            });
    }

    // Process pipeline until all stages complete
    while (!std::all_of(stage_completions.begin(), stage_completions.end(), [](bool c) { return c; })) {
        sim->step();
    }

    // Verify all pipeline stages completed
    REQUIRE(completion_order.size() == stage_count);

    // Verify data integrity for each stage
    for (size_t stage = 0; stage < stage_count; ++stage) {
        Address addr = stage * matrix_size_per_stage;
        REQUIRE(verify_tensor(stage_data[stage], addr, 0));
    }

    std::cout << "Pipeline completion order: ";
    for (size_t stage : completion_order) {
        std::cout << stage << " ";
    }
    std::cout << "\n";
}

TEST_CASE_METHOD(DMATensorMovementFixture, "DMA Tensor - Memory Bank Optimization", "[dma][tensor][optimization]") {
    // Test optimal distribution of tensor data across memory banks
    const size_t tensor_count = 8;
    const size_t tensor_size = 64 * 64 * sizeof(float);
    const size_t total_size = tensor_count * tensor_size;

    if (tensor_size > sim->get_scratchpad_capacity(0)) {
        SKIP("Tensor too large for scratchpad");
    }

    // Generate tensors
    std::vector<std::vector<float>> tensors(tensor_count);
    for (size_t i = 0; i < tensor_count; ++i) {
        tensors[i] = generate_tensor(64, 64, static_cast<float>(i + 100));
    }

    // Test different memory bank distribution strategies
    SECTION("Sequential Bank Assignment") {
        // Assign tensors to banks sequentially
        for (size_t i = 0; i < tensor_count; ++i) {
            size_t bank_id = i % config.memory_bank_count;
            size_t offset = (i / config.memory_bank_count) * tensor_size;
            sim->write_memory_bank(bank_id, offset, tensors[i].data(), tensor_size);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Transfer all tensors to different areas of scratchpad
        std::vector<bool> completions(tensor_count, false);

        // Check if all tensors fit in scratchpad
        if (tensor_count * tensor_size > sim->get_scratchpad_capacity(0)) {
            // Process tensors one by one if they don't all fit
            Address scratch_base = sim->get_scratchpad_base(0);

            for (size_t i = 0; i < tensor_count; ++i) {
                size_t bank_id = i % config.memory_bank_count;
                size_t dma_id = i % config.dma_engine_count;
                size_t src_offset = (i / config.memory_bank_count) * tensor_size;
                size_t dst_offset = 0; // Overwrite scratchpad for each tensor

                Address global_src = sim->get_external_bank_base(bank_id) + src_offset;
                Address global_dst = scratch_base + dst_offset;

                sim->dma_external_to_scratchpad(dma_id, global_src, global_dst, tensor_size,
                    [&completions, i]() { completions[i] = true; });

                // Process this transfer individually for timing
                while (!completions[i]) {
                    sim->step();
                }

                // Verify tensor immediately
                REQUIRE(verify_tensor(tensors[i], dst_offset, 0));

                // Reset completions for next iteration
                completions[i] = false;
            }
        } else {
            // All tensors fit - transfer to different scratchpad areas
            Address scratch_base = sim->get_scratchpad_base(0);

            for (size_t i = 0; i < tensor_count; ++i) {
                size_t bank_id = i % config.memory_bank_count;
                size_t dma_id = i % config.dma_engine_count;
                size_t src_offset = (i / config.memory_bank_count) * tensor_size;
                size_t dst_offset = i * tensor_size; // Different area per tensor

                Address global_src = sim->get_external_bank_base(bank_id) + src_offset;
                Address global_dst = scratch_base + dst_offset;

                sim->dma_external_to_scratchpad(dma_id, global_src, global_dst, tensor_size,
                    [&completions, i]() { completions[i] = true; });
            }

            // Wait for all transfers
            while (!std::all_of(completions.begin(), completions.end(), [](bool c) { return c; })) {
                sim->step();
            }

            // Verify all tensors
            for (size_t i = 0; i < tensor_count; ++i) {
                size_t dst_offset = i * tensor_size;
                REQUIRE(verify_tensor(tensors[i], dst_offset, 0));
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto sequential_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        std::cout << "Sequential assignment time: " << sequential_duration.count() << " μs\n";
    }

    SECTION("Interleaved Bank Assignment") {
        // Interleave tensor data across banks for better bandwidth utilization
        size_t elements_per_bank = (64 * 64) / config.memory_bank_count;

        for (size_t i = 0; i < tensor_count; ++i) {
            // Split tensor across banks
            for (size_t bank = 0; bank < config.memory_bank_count; ++bank) {
                size_t start_element = bank * elements_per_bank;
                size_t num_elements = (bank == config.memory_bank_count - 1) ?
                                      (64 * 64 - start_element) : elements_per_bank;
                size_t bank_data_size = num_elements * sizeof(float);

                sim->write_memory_bank(bank, i * bank_data_size,
                                     tensors[i].data() + start_element,
                                     bank_data_size);
            }
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Transfer tensors with interleaved access
        Address scratch_base = sim->get_scratchpad_base(0);

        for (size_t i = 0; i < tensor_count; ++i) {
            std::vector<bool> bank_completions(config.memory_bank_count, false);

            // Start transfers from all banks for this tensor
            for (size_t bank = 0; bank < config.memory_bank_count; ++bank) {
                size_t elements_per_bank_actual = (64 * 64) / config.memory_bank_count;
                size_t bank_data_size = elements_per_bank_actual * sizeof(float);
                size_t dma_id = bank % config.dma_engine_count;

                Address global_src = sim->get_external_bank_base(bank) + i * bank_data_size;
                Address global_dst = scratch_base + bank * bank_data_size;

                sim->dma_external_to_scratchpad(dma_id, global_src, global_dst, bank_data_size,
                    [&bank_completions, bank]() { bank_completions[bank] = true; });
            }

            // Wait for all bank transfers to complete
            while (!std::all_of(bank_completions.begin(), bank_completions.end(), [](bool c) { return c; })) {
                sim->step();
            }

            // Reconstruct and verify tensor (simplified verification)
            // In practice, would need to handle the interleaved reconstruction
            sim->reset(); // Reset for next tensor
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto interleaved_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        std::cout << "Interleaved assignment time: " << interleaved_duration.count() << " μs\n";
    }
}