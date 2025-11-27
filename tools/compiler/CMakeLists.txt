# ============================================================================
# tools/compiler/CMakeLists.txt
# KPU Compilation Toolchain
# ============================================================================

# KPU Kernel Compiler
# Compiles Domain Flow Graphs (.dfg) to KPU object files (.kpu)
kpu_add_tool(kernel-compiler
    FOLDER "Compiler"
    DESCRIPTION "KPU kernel compiler - compiles DFG to KPU object files"
    SOURCES
        kpu-kernel-compiler/main.cpp
        kpu-kernel-compiler/dfg_parser.cpp
        kpu-kernel-compiler/kir_generator.cpp
        kpu-kernel-compiler/object_writer.cpp
    LIBS
        nlohmann_json::nlohmann_json
        kpu_compiler
)

# Add include directories for compiler components
target_include_directories(kernel-compiler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kpu-kernel-compiler
)
