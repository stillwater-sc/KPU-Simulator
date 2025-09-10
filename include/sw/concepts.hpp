pragma once
// core types and concepts for KPU simulator
#include <cstdint>
#include <cstddef>

namespace sw::kpu {

// Base address types
using Address = std::uint64_t;
using Size = std::size_t;
using Cycle = std::uint64_t;

}