#!/bin/bash

echo "========================================"
echo "KPU Simulator - Development Environment Setup"
echo "========================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check prerequisites
echo "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_status "Python found: $PYTHON_VERSION"

# Check Python version (3.8+)
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"
if [ $? -ne 0 ]; then
    print_error "Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi

# Check CMake
if ! command -v cmake &> /dev/null; then
    print_error "CMake not found. Please install CMake 3.16+"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
print_status "CMake found: $CMAKE_VERSION"

# Check compiler
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1)
    print_status "Compiler found: $GCC_VERSION"
elif command -v clang++ &> /dev/null; then
    CLANG_VERSION=$(clang++ --version | head -n1)
    print_status "Compiler found: $CLANG_VERSION"
else
    print_warning "C++ compiler not found. You may need to install build tools."
fi

echo
echo "Setting up development environment..."

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clean up existing venv if it exists
if [ -d "$PROJECT_DIR/kpu_venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$PROJECT_DIR/kpu_venv"
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv "$PROJECT_DIR/kpu_venv"
if [ $? -ne 0 ]; then
    print_error "Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$PROJECT_DIR/kpu_venv/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "Installing Python dependencies from requirements.txt..."
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo "Installing basic Python dependencies..."
    pip install "pybind11[global]" numpy matplotlib seaborn jupyter ipywidgets
fi

if [ $? -ne 0 ]; then
    print_error "Failed to install Python dependencies"
    exit 1
fi

# Verify installation
echo "Verifying installation..."
python -c "import pybind11; print('✓ pybind11:', pybind11.__version__)" || {
    print_error "pybind11 verification failed"
    exit 1
}

python -c "import numpy; print('✓ numpy:', numpy.__version__)" || {
    print_error "numpy verification failed"
    exit 1
}

echo
echo "========================================"
echo "Development Environment Setup Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Configure and build the C++ project:"
echo "   mkdir build && cd build"
echo "   cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "   make -j\$(nproc)"
echo
echo "2. Test Python bindings:"
echo "   source kpu_venv/bin/activate"
echo "   python tests/test_python.py"
echo
echo "3. Start Jupyter for interactive development:"
echo "   source kpu_venv/bin/activate"
echo "   jupyter notebook"
echo
echo "Environment details:"
echo "  Project: $PROJECT_DIR"
echo "  Virtual env: $PROJECT_DIR/kpu_venv"
echo "  Python: $PYTHON_VERSION"
echo "  CMake: $CMAKE_VERSION"
echo
