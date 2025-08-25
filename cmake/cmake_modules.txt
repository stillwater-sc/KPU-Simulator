# cmake/Packaging.cmake
# Package configuration for distribution

include(GNUInstallDirs)

# CPack configuration
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VENDOR "Stillwater Supercomputing, Inc.")
set(CPACK_PACKAGE_CONTACT "info@stillwater-sc.com")
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_SOURCE_DIR}/LICENSE)
set(CPACK_RESOURCE_FILE_README ${CMAKE_SOURCE_DIR}/README.md)

# Platform-specific packaging
if(WIN32)
    # Windows - NSIS installer
    set(CPACK_GENERATOR "NSIS;ZIP")
    set(CPACK_NSIS_DISPLAY_NAME "Stillwater KPU Simulator")
    set(CPACK_NSIS_PACKAGE_NAME "StillwaterKPU")
    set(CPACK_NSIS_CONTACT ${CPACK_PACKAGE_CONTACT})
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    
elseif(APPLE)
    # macOS - DMG and Bundle
    set(CPACK_GENERATOR "DragNDrop;TGZ")
    set(CPACK_DMG_FORMAT "UDBZ")
    set(CPACK_DMG_VOLUME_NAME "Stillwater KPU Simulator")
    
else()
    # Linux - DEB and RPM packages
    set(CPACK_GENERATOR "DEB;RPM;TGZ")
    
    # DEB package configuration
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT})
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libstdc++6, libgomp1")
    
    # RPM package configuration
    set(CPACK_RPM_PACKAGE_GROUP "Applications/Engineering")
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_PACKAGE_REQUIRES "glibc, libstdc++, libgomp")
endif()

# Component-based packaging
set(CPACK_COMPONENTS_ALL Runtime Development Tools Python Documentation)

# Runtime component
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "Runtime Libraries")
set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "Core KPU simulator runtime libraries")
set(CPACK_COMPONENT_RUNTIME_REQUIRED ON)

# Development component
set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "Development Files")
set(CPACK_COMPONENT_DEVELOPMENT_DESCRIPTION "Headers and libraries for development")
set(CPACK_COMPONENT_DEVELOPMENT_DEPENDS Runtime)

# Tools component
set(CPACK_COMPONENT_TOOLS_DISPLAY_NAME "Development Tools")
set(CPACK_COMPONENT_TOOLS_DESCRIPTION "Command-line tools and utilities")
set(CPACK_COMPONENT_TOOLS_DEPENDS Runtime)

# Python component
if(KPU_BUILD_PYTHON_BINDINGS)
    set(CPACK_COMPONENT_PYTHON_DISPLAY_NAME "Python Bindings")
    set(CPACK_COMPONENT_PYTHON_DESCRIPTION "Python API and tools")
    set(CPACK_COMPONENT_PYTHON_DEPENDS Runtime)
endif()

# Documentation component
if(KPU_BUILD_DOCS)
    set(CPACK_COMPONENT_DOCUMENTATION_DISPLAY_NAME "Documentation")
    set(CPACK_COMPONENT_DOCUMENTATION_DESCRIPTION "API documentation and examples")
endif()

include(CPack)
