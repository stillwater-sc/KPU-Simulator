/**
 * @file object_writer.cpp
 * @brief Implementation of KPU object file writer
 */

#include "object_writer.hpp"
#include <fstream>
#include <stdexcept>

namespace sw::kpu::compiler {

ObjectWriter::ObjectWriter(const ObjectWriterOptions& options)
    : options_(options)
{
}

void ObjectWriter::write(const kir::Program& program, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    write(program, file);
}

void ObjectWriter::write(const kir::Program& program, std::ostream& os) {
    os << to_string(program);
}

std::string ObjectWriter::to_string(const kir::Program& program) {
    kir::json j = kir::program_to_json(program);

    // Remove labels if not requested
    if (!options_.include_labels) {
        for (auto& op : j["operations"]) {
            op.erase("label");
        }
    }

    // Remove hints if not requested
    if (!options_.include_hints) {
        j.erase("hints");
    }

    if (options_.pretty_print) {
        return j.dump(2);  // 2-space indentation
    } else {
        return j.dump();
    }
}

} // namespace sw::kpu::compiler
