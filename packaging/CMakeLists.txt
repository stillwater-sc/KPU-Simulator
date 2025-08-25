# ============================================================================
# packaging/CMakeLists.txt
# Package creation and distribution

# Docker configuration
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/docker/Dockerfile.dev.in
    ${CMAKE_CURRENT_BINARY_DIR}/docker/Dockerfile.dev
    @ONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/docker/Dockerfile.runtime.in
    ${CMAKE_CURRENT_BINARY_DIR}/docker/Dockerfile.runtime
    @ONLY
)

# Install packaging files
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docker/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/stillwater-kpu/docker
    COMPONENT Development
)
