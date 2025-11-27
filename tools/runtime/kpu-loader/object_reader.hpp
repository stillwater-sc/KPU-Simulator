/**
 * @file object_reader.hpp
 * @brief KPU object file reader
 */

#pragma once

#include <sw/compiler/kir/kir.hpp>
#include <sw/compiler/kir/object_file.hpp>
#include <string>

namespace sw::kpu::runtime {

/**
 * @brief Reads KPU object files
 */
class ObjectReader {
public:
    /**
     * @brief Read a KIR program from an object file
     *
     * @param filename Input filename (.kpu)
     * @return Loaded KIR program
     */
    kir::Program read(const std::string& filename);

    /**
     * @brief Validate a loaded program
     *
     * @param program Program to validate
     * @return true if valid, false otherwise
     */
    bool validate(const kir::Program& program);

    /**
     * @brief Get validation errors
     */
    const std::vector<std::string>& errors() const { return errors_; }

private:
    std::vector<std::string> errors_;
};

} // namespace sw::kpu::runtime
