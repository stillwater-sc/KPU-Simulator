#!/bin/bash
# scripts/build.sh
# Main build script for Stillwater KPU project

set -e

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build"
INSTALL_PREFIX="/usr/local"
PARALLEL_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Features
BUILD_TESTS=ON
BUILD_EXAMPLES=ON
BUILD_TOOLS=ON
BUILD_PYTHON=ON
BUILD_BENCHMARKS=ON
BUILD_DOCS=OFF
ENABLE_OPENMP=ON
ENABLE_CUDA=OFF
ENABLE_OPENCL=OFF
ENABLE_SANITIZERS=OFF
ENABLE_PROFILING=OFF
STATIC_ANALYSIS=OFF

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_help() {
    cat << EOF
Stillwater KPU Simulator Build Script

Usage: $0 [OPTIONS]

Options:
  -h, --help              Show this help message
  -t, --type TYPE         Build type: Debug, Release, RelWithDebInfo, MinSizeRel (default: Release)
  -d, --build-dir DIR     Build directory (default: build)
  -p, --prefix PREFIX     Install prefix (default: /usr/local)
  -j, --jobs JOBS         Parallel build jobs (default: auto-detected)
  
  --no-tests              Disable test building
  --no-examples           Disable examples building
  --no-tools              Disable tools building  
  --no-python             Disable Python bindings
  --no-benchmarks         Disable benchmarks
  --docs                  Enable documentation building
  
  --cuda                  Enable CUDA support
  --opencl                Enable OpenCL support
  --no-openmp             Disable OpenMP support
  
  --sanitizers            Enable sanitizers (Debug builds)
  --profiling             Enable profiling support
  --static-analysis       Enable static analysis tools
  
  --clean                 Clean build directory first
  --install               Install after building
  --package               Create packages after building
  --verbose               Verbose build output

Examples:
  $0                                    # Basic release build
  $0 --type Debug --sanitizers         # Debug build with sanitizers
  $0 --cuda --opencl --docs --install  # Full featured build with installation
  $0 --clean --package                 # Clean build and create packages
EOF
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking build dependencies..."
    
    # Check CMake
    if ! command -v cmake &> /dev/null; then
        log_error "CMake is required but not installed"
        exit 1
    fi
    
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    CMAKE_MAJOR=$(echo $CMAKE_VERSION | cut -d'.' -f1)
    CMAKE_MINOR=$(echo $CMAKE_VERSION | cut -d'.' -f2)
    
    if [ $CMAKE_MAJOR -lt 3 ] || [ $CMAKE_MAJOR -eq 3 -a $CMAKE_MINOR -lt 18 ]; then
        log_error "CMake 3.18 or newer is required (found $CMAKE_VERSION)"
        exit 1
    fi
    
    log_success "CMake $CMAKE_VERSION found"
    
    # Check C++ compiler
    if command -v g++ &> /dev/null; then
        GCC_VERSION=$(g++ --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
        log_success "GCC $GCC_VERSION found"
    elif command -v clang++ &> /dev/null; then
        CLANG_VERSION=$(clang++ --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
        log_success "Clang $CLANG_VERSION found"
    else
        log_error "No suitable C++ compiler found"
        exit 1
    fi
    
    # Check Python if building bindings
    if [ "$BUILD_PYTHON" = "ON" ]; then
        if command -v python3 &> /dev/null; then
            PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
            log_success "Python $PYTHON_VERSION found"
        else
            log_warning "Python3 not found, disabling Python bindings"
            BUILD_PYTHON=OFF
        fi
    fi
}

configure_build() {
    log_info "Configuring build in $BUILD_DIR..."
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # CMake arguments
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
        -DKPU_BUILD_TESTS="$BUILD_TESTS"
        -DKPU_BUILD_EXAMPLES="$BUILD_EXAMPLES"
        -DKPU_BUILD_TOOLS="$BUILD_TOOLS"
        -DKPU_BUILD_PYTHON_BINDINGS="$BUILD_PYTHON"
        -DKPU_BUILD_BENCHMARKS="$BUILD_BENCHMARKS"
        -DKPU_BUILD_DOCS="$BUILD_DOCS"
        -DKPU_ENABLE_OPENMP="$ENABLE_OPENMP"
        -DKPU_ENABLE_CUDA="$ENABLE_CUDA"
        -DKPU_ENABLE_OPENCL="$ENABLE_OPENCL"
        -DKPU_ENABLE_SANITIZERS="$ENABLE_SANITIZERS"
        -DKPU_ENABLE_PROFILING="$ENABLE_PROFILING"
        -DKPU_STATIC_ANALYSIS="$STATIC_ANALYSIS"
    )
    
    # Add verbose output if requested
    if [ "$VERBOSE" = "true" ]; then
        CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
    fi
    
    # Run CMake configuration
    if cmake "${CMAKE_ARGS[@]}" ..; then
        log_success "Configuration completed successfully"
    else
        log_error "Configuration failed"
        exit 1
    fi
}

build_project() {
    log_info "Building project with $PARALLEL_JOBS parallel jobs..."
    
    BUILD_ARGS=(--build . --parallel "$PARALLEL_JOBS")
    
    if [ "$VERBOSE" = "true" ]; then
        BUILD_ARGS+=(--verbose)
    fi
    
    if cmake "${BUILD_ARGS[@]}"; then
        log_success "Build completed successfully"
    else
        log_error "Build failed"
        exit 1
    fi
}

run_tests() {
    if [ "$BUILD_TESTS" = "ON" ]; then
        log_info "Running tests..."
        if ctest --output-on-failure --parallel "$PARALLEL_JOBS"; then
            log_success "All tests passed"
        else
            log_error "Some tests failed"
            return 1
        fi
    fi
}

install_project() {
    log_info "Installing project to $INSTALL_PREFIX..."
    if cmake --install .; then
        log_success "Installation completed successfully"
    else
        log_error "Installation failed"
        exit 1
    fi
}

create_packages() {
    log_info "Creating packages..."
    if cpack; then
        log_success "Packages created successfully"
        ls -la *.deb *.rpm *.tar.gz *.dmg *.zip 2>/dev/null || true
    else
        log_error "Package creation failed"
        exit 1
    fi
}

# Parse command line arguments
CLEAN_BUILD=false
INSTALL_AFTER=false
CREATE_PACKAGES=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_help
            exit 0
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -d|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -p|--prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --no-tests)
            BUILD_TESTS=OFF
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES=OFF
            shift
            ;;
        --no-tools)
            BUILD_TOOLS=OFF
            shift
            ;;
        --no-python)
            BUILD_PYTHON=OFF
            shift
            ;;
        --no-benchmarks)
            BUILD_BENCHMARKS=OFF
            shift
            ;;
        --docs)
            BUILD_DOCS=ON
            shift
            ;;
        --cuda)
            ENABLE_CUDA=ON
            shift
            ;;
        --opencl)
            ENABLE_OPENCL=ON
            shift
            ;;
        --no-openmp)
            ENABLE_OPENMP=OFF
            shift
            ;;
        --sanitizers)
            ENABLE_SANITIZERS=ON
            shift
            ;;
        --profiling)
            ENABLE_PROFILING=ON
            shift
            ;;
        --static-analysis)
            STATIC_ANALYSIS=ON
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --install)
            INSTALL_AFTER=true
            shift
            ;;
        --package)
            CREATE_PACKAGES=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_info "Starting Stillwater KPU build process..."
    log_info "Build type: $BUILD_TYPE"
    log_info "Build directory: $BUILD_DIR"
    log_info "Install prefix: $INSTALL_PREFIX"
    log_info "Parallel jobs: $PARALLEL_JOBS"
    
    # Check dependencies
    check_dependencies
    
    # Clean build directory if requested
    if [ "$CLEAN_BUILD" = "true" ] && [ -d "$BUILD_DIR" ]; then
        log_info "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
        log_success "Build directory cleaned"
    fi
    
    # Configure and build
    configure_build
    build_project
    
    # Run tests
    run_tests
    
    # Install if requested
    if [ "$INSTALL_AFTER" = "true" ]; then
        install_project
    fi
    
    # Create packages if requested
    if [ "$CREATE_PACKAGES" = "true" ]; then
        create_packages
    fi
    
    log_success "Build process completed successfully!"
    
    # Print summary
    echo
    echo "================================"
    echo "Build Summary:"
    echo "  Type: $BUILD_TYPE"
    echo "  Directory: $BUILD_DIR"
    echo "  Features:"
    echo "    Tests: $BUILD_TESTS"
    echo "    Examples: $BUILD_EXAMPLES" 
    echo "    Tools: $BUILD_TOOLS"
    echo "    Python: $BUILD_PYTHON"
    echo "    OpenMP: $ENABLE_OPENMP"
    echo "    CUDA: $ENABLE_CUDA"
    echo "    OpenCL: $ENABLE_OPENCL"
    echo "================================"
}

# Run main function
main "$@"