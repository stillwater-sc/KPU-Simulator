# ============================================================================
# tools/configuration/CMakeLists.txt
# KPU Configuration Tools
# ============================================================================

# KPU Config Tool
kpu_add_tool(config
    FOLDER "Configuration"
    DESCRIPTION "KPU configuration and validation tool"
    SOURCES
        kpu-config/kpu_config.cpp
    LIBS
        nlohmann_json::nlohmann_json
)
