/**
 * @file object_writer.hpp
 * @brief KPU object file writer
 *
 * Writes DFX programs to .kpu object files that can be loaded
 * by the KPU loader/driver.
 */

#pragma once

#include <sw/compiler/dfx/dfx.hpp>
#include <sw/compiler/dfx/dfx_object_file.hpp>
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
 * @brief Writes DFX programs to object files
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
     * @param program DFX program to write
     * @param filename Output filename (should end in .kpu)
     */
    void write(const dfx::Program& program, const std::string& filename);

    /**
     * @brief Write program to output stream
     *
     * @param program DFX program to write
     * @param os Output stream
     */
    void write(const dfx::Program& program, std::ostream& os);

    /**
     * @brief Get program as JSON string
     *
     * @param program DFX program to serialize
     * @return JSON string representation
     */
    std::string to_string(const dfx::Program& program);

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
