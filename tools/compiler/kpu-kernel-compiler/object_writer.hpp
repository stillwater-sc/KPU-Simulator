/**
 * @file object_writer.hpp
 * @brief KPU object file writer
 *
 * Writes KIR programs to .kpu object files that can be loaded
 * by the KPU loader/driver.
 */

#pragma once

#include <sw/compiler/kir/kir.hpp>
#include <sw/compiler/kir/object_file.hpp>
#include <string>
#include <ostream>

namespace sw::kpu::compiler {

/**
 * @brief Options for object file writing
 */
struct ObjectWriterOptions {
    bool pretty_print = true;       ///< Pretty print JSON (human-readable)
    bool include_hints = true;      ///< Include performance hints
    bool include_labels = true;     ///< Include operation labels (for debugging)
};

/**
 * @brief Writes KIR programs to object files
 */
class ObjectWriter {
public:
    /**
     * @brief Constructor
     * @param options Writer options
     */
    explicit ObjectWriter(const ObjectWriterOptions& options = ObjectWriterOptions());

    /**
     * @brief Write program to file
     *
     * @param program KIR program to write
     * @param filename Output filename (should end in .kpu)
     */
    void write(const kir::Program& program, const std::string& filename);

    /**
     * @brief Write program to output stream
     *
     * @param program KIR program to write
     * @param os Output stream
     */
    void write(const kir::Program& program, std::ostream& os);

    /**
     * @brief Get program as JSON string
     *
     * @param program KIR program to serialize
     * @return JSON string representation
     */
    std::string to_string(const kir::Program& program);

    /**
     * @brief Set writer options
     */
    void set_options(const ObjectWriterOptions& options) { options_ = options; }

    /**
     * @brief Get current options
     */
    const ObjectWriterOptions& options() const { return options_; }

private:
    ObjectWriterOptions options_;
};

} // namespace sw::kpu::compiler
