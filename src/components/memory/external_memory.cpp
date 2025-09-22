#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <sw/memory/external_memory.hpp>

namespace sw::kpu {

// ExternalMemory implementation - manages its own memory model
ExternalMemory::ExternalMemory(Size capacity_mb, Size bandwidth_gbps)
    : capacity(capacity_mb * 1024 * 1024),
      bandwidth_bytes_per_cycle(bandwidth_gbps * 1000000000 / 8 / 1000000000), // Assuming 1GHz clock
      last_access_cycle(0) {
    memory_model.resize(capacity);
    std::fill(memory_model.begin(), memory_model.end(), static_cast<uint8_t>(0));
}

void ExternalMemory::read(Address addr, void* data, Size size) {
    if (addr + size > capacity) {
        throw std::out_of_range("Memory read out of bounds");
    }
    std::memcpy(data, memory_model.data() + addr, size);
}

void ExternalMemory::write(Address addr, const void* data, Size size) {
    if (addr + size > capacity) {
        throw std::out_of_range("Memory write out of bounds");
    }
    std::memcpy(memory_model.data() + addr, data, size);
}

bool ExternalMemory::is_ready() const {
    // Simplified: assume always ready for now
    return true;
}

void ExternalMemory::reset() {
    std::fill(memory_model.begin(), memory_model.end(), static_cast<uint8_t>(0));
    last_access_cycle = 0;
}

} // namespace sw::kpu