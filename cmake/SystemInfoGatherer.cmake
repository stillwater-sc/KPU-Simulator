# cmake/SystemInfoGatherer.cmake

# Set the output log file path in the build directory
set(SYSTEM_INFO_LOG_FILE "${CMAKE_BINARY_DIR}/build_system_info.log")

# Clear the file before writing new information
file(WRITE ${SYSTEM_INFO_LOG_FILE} "--- CMake Build System Info ---\n")

# 1. Gather Universal Information
file(APPEND ${SYSTEM_INFO_LOG_FILE} "Date: ${CURRENT_DATE}\n") # Note: CURRENT_DATE must be set if needed
file(APPEND ${SYSTEM_INFO_LOG_FILE} "CMake Version: ${CMAKE_VERSION}\n")
file(APPEND ${SYSTEM_INFO_LOG_FILE} "System Name: ${CMAKE_SYSTEM_NAME}\n")
file(APPEND ${SYSTEM_INFO_LOG_FILE} "Processor: ${CMAKE_SYSTEM_PROCESSOR}\n")
file(APPEND ${SYSTEM_INFO_LOG_FILE} "Compiler (C): ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}\n")
file(APPEND ${SYSTEM_INFO_LOG_FILE} "Compiler Flags (C): ${CMAKE_C_FLAGS}\n")
file(APPEND ${SYSTEM_INFO_LOG_FILE} "---------------------------------\n")
message(STATUS "Gathering platform-specific runtime information...")

# 2. Platform-Specific Runtime/Library Identification

# --- LINUX (Likely Glibc) ---
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    file(APPEND ${SYSTEM_INFO_LOG_FILE} "--- Linux C Library (Likely Glibc) Info ---\n")

    # Try common package managers for Glibc version
    execute_process(COMMAND dpkg -s libc6 OUTPUT_VARIABLE PKG_OUTPUT ERROR_QUIET RESULT_VARIABLE PKG_RESULT)
    if (PKG_RESULT EQUAL 0)
        file(APPEND ${SYSTEM_INFO_LOG_FILE} "libc6 (dpkg):\n${PKG_OUTPUT}\n")
        message(STATUS "Captured libc6 info via dpkg.")
    else()
        execute_process(COMMAND rpm -qi glibc OUTPUT_VARIABLE PKG_OUTPUT ERROR_QUIET RESULT_VARIABLE PKG_RESULT)
        if (PKG_RESULT EQUAL 0)
            file(APPEND ${SYSTEM_INFO_LOG_FILE} "glibc (rpm):\n${PKG_OUTPUT}\n")
            message(STATUS "Captured glibc info via rpm.")
        else()
            file(APPEND ${SYSTEM_INFO_LOG_FILE} "Glibc Info: Package manager check failed. Check OS version manually.\n")
            message(STATUS "Glibc info not found via common package managers.")
        endif()
    endif()

# --- WINDOWS (MSVC, MinGW, or Clang/LLVM) ---
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    file(APPEND ${SYSTEM_INFO_LOG_FILE} "--- Windows Runtime Library Info ---\n")

    if (CMAKE_C_COMPILER_ID MATCHES "MSVC")
        # For MSVC, the runtime version is tied to the Visual Studio version
        file(APPEND ${SYSTEM_INFO_LOG_FILE} "MSVC Runtime (MSVCRT): Version is tied to Visual Studio ${MSVC_VERSION}\n")
        file(APPEND ${SYSTEM_INFO_LOG_FILE} "Runtime Linkage Type: (Check CMAKE_C_FLAGS for /MD, /MT, etc.)\n")
        message(STATUS "Windows platform using MSVC.")
    elseif (CMAKE_C_COMPILER_ID MATCHES "GNU")
        # For MinGW/GCC, the runtime is MinGW's implementation of the C library
        file(APPEND ${SYSTEM_INFO_LOG_FILE} "MinGW Runtime: Using MinGW-w64/MinGW runtime.\n")
        # The MinGW version is harder to reliably query than Glibc/package managers
        message(STATUS "Windows platform using MinGW/GCC.")
    else()
        file(APPEND ${SYSTEM_INFO_LOG_FILE} "Compiler ID: ${CMAKE_C_COMPILER_ID}. C Runtime version is platform-dependent.\n")
    endif()
    
# --- MACOS (Clang/LLVM and Darwin) ---
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    file(APPEND ${SYSTEM_INFO_LOG_FILE} "--- macOS C Library Info (Darwin/libSystem) ---\n")

    # On macOS, the C standard library (including libm) is part of libSystem.dylib 
    # which is managed by the OS and Xcode's command line tools.
    
    # Record the OS version (major difference in macOS could mean different libm)
    execute_process(COMMAND sw_vers -productVersion OUTPUT_VARIABLE OS_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    file(APPEND ${SYSTEM_INFO_LOG_FILE} "macOS Version: ${OS_VERSION}\n")

    # Record the Xcode Command Line Tools version
    execute_process(COMMAND xcode-select -p OUTPUT_VARIABLE XCODE_PATH ERROR_QUIET)
    if (XCODE_PATH)
        execute_process(COMMAND xcrun -f clang OUTPUT_VARIABLE CLANG_PATH ERROR_QUIET)
        file(APPEND ${SYSTEM_INFO_LOG_FILE} "Xcode CLT Path: ${XCODE_PATH}\n")
        # The version of the libm is tied to the Darwin kernel and the linked SDK.
        file(APPEND ${SYSTEM_INFO_LOG_FILE} "C Library: Provided by Darwin/libSystem.dylib (Version tied to OS and SDK).\n")
    else()
        file(APPEND ${SYSTEM_INFO_LOG_FILE} "C Library: Provided by Darwin/libSystem.dylib. Xcode Command Line Tools not found.\n")
    endif()

# --- OTHER PLATFORMS ---
else()
    file(APPEND ${SYSTEM_INFO_LOG_FILE} "--- Unknown C Library Info ---\n")
    file(APPEND ${SYSTEM_INFO_LOG_FILE} "Platform: ${CMAKE_SYSTEM_NAME}. Manual investigation of the C runtime is required.\n")

endif()

file(APPEND ${SYSTEM_INFO_LOG_FILE} "---------------------------------\n")
message(STATUS "System information logged to: ${SYSTEM_INFO_LOG_FILE}")