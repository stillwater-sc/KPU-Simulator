# cmake/Dependencies.cmake
# External dependency management

include(FetchContent)

# Set common FetchContent properties
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# Function to add external dependency
function(kpu_add_dependency name)
    set(options REQUIRED OPTIONAL)
    set(oneValueArgs GIT_REPOSITORY GIT_TAG CMAKE_ARGS)
    set(multiValueArgs TARGETS)
    cmake_parse_arguments(DEP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    if(NOT TARGET ${name})
        message(STATUS "Adding dependency: ${name}")
        
        FetchContent_Declare(${name}
            GIT_REPOSITORY ${DEP_GIT_REPOSITORY}
            GIT_TAG ${DEP_GIT_TAG}
            ${DEP_CMAKE_ARGS}
        )
        
        FetchContent_MakeAvailable(${name})
        
        # Set folder for IDE organization
        if(DEP_TARGETS)
            foreach(target ${DEP_TARGETS})
                if(TARGET ${target})
                    set_target_properties(${target} PROPERTIES
                        FOLDER "Third Party/${name}"
                    )
                endif()
            endforeach()
        endif()
    endif()
endfunction()

# Define common dependencies
if(KPU_BUILD_TESTS)
    kpu_add_dependency(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
        TARGETS Catch2 Catch2WithMain
    )
endif()

kpu_add_dependency(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
    TARGETS spdlog spdlog_header_only
)

kpu_add_dependency(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
    TARGETS nlohmann_json
)

kpu_add_dependency(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.1.1
    TARGETS fmt fmt-header-only
)

if(KPU_BUILD_PYTHON_BINDINGS)
    kpu_add_dependency(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
        TARGETS pybind11 pybind11_headers
    )
endif()

