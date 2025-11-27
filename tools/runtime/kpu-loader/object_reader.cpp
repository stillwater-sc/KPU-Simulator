/**
 * @file object_reader.cpp
 * @brief Implementation of KPU object file reader
 */

#include "object_reader.hpp"

namespace sw::kpu::runtime {

using namespace sw::kpu::compiler;

kir::Program ObjectReader::read(const std::string& filename) {
    errors_.clear();
    return kir::read_object_file(filename);
}

bool ObjectReader::validate(const kir::Program& program) {
    errors_.clear();

    // Check version compatibility
    if (program.version_major > kir::KIR_VERSION_MAJOR) {
        errors_.push_back("Incompatible KIR version: " +
                         std::to_string(program.version_major) + "." +
                         std::to_string(program.version_minor));
        return false;
    }

    // Check that we have tensors
    if (program.tensors.empty()) {
        errors_.push_back("Program has no tensors");
        return false;
    }

    // Check that we have operations
    if (program.operations.empty()) {
        errors_.push_back("Program has no operations");
        return false;
    }

    // Check tiling configuration
    if (program.tiling.Ti == 0 || program.tiling.Tj == 0 || program.tiling.Tk == 0) {
        errors_.push_back("Invalid tiling configuration");
        return false;
    }

    return true;
}

} // namespace sw::kpu::runtime
