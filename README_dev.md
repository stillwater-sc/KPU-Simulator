# KPU Simulator - Developer Setup

This guide helps new contributors set up their development environment for the KPU Simulator project.

## Prerequisites

### All Platforms
- **Python 3.8+** with pip
- **CMake 3.16+**
- **C++20 compatible compiler**

### Windows Specific
- **Visual Studio 2019+** or **Visual Studio Build Tools**
- **Git for Windows** (recommended)

### Linux Specific
- **GCC 10+** or **Clang 10+**
- **Build essentials**: `sudo apt install build-essential cmake git`

### macOS Specific  
- **Xcode Command Line Tools**: `xcode-select --install`
- **Homebrew** (recommended): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

## Quick Setup

### Windows
```batch
git clone <repository-url>
cd KPU-simulator
dev_setup_windows.bat
```

### Linux/macOS
```bash
git clone <repository-url>
cd KPU-simulator
chmod +x setup_development_environment.sh
./dev_setup_unix.sh
```

## Manual Setup (if scripts fail)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd KPU-simulator
   ```

2. **Create virtual environment**
   ```bash
   # Windows
   python -m venv kpu_venv
   kpu_venv\Scripts\activate
   
   # Linux/macOS
   python3 -m venv kpu_venv
   source kpu_venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure and build**
   ```bash
   # Windows (with CMake GUI)
   cmake_gui_setup_fixed.bat
   
   # Command line (all platforms)
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release
   ```

## Project Structure

```
KPU-simulator/
├── include/sw/kpu/          # C++ headers
├── src/
│   ├── simulator/           # C++ implementation
│   └── bindings/python/     # Python bindings
├── tests/                   # Test files
├── tools/python/            # Python visualization tools
├── kpu_venv/                # Virtual environment (not in git)
├── build/                   # Build directory (not in git)
├── requirements.txt         # Python dependencies
└── README.md
```

## Development Workflow

### Daily Development
```bash
# Activate environment
source kpu_venv/bin/activate  # Linux/macOS
kpu_venv\Scripts\activate     # Windows

# Build after C++ changes
cmake --build build --config Release

# Test Python bindings
python tests/test_python.py

# Interactive development
jupyter notebook tools/python/kpu_demo.ipynb
```

### Testing
```bash
# C++ tests
build/Release/kpu_test.exe    # Windows
build/kpu_test               # Linux/macOS

# Python tests
python tests/test_python.py

# Python visualization tests
python tools/python/kpu_visualizer.py
```

## Troubleshooting

### Python Virtual Environment Issues
```bash
# Delete and recreate
rm -rf kpu_venv
python -m venv kpu_venv
# Activate environment
source kpu_venv/bin/activate  # Linux/macOS
kpu_venv\Scripts\activate     # Windows
# ... reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### CMake Can't Find Python/pybind11
```bash
# Specify paths explicitly
cmake .. -DPython_ROOT_DIR="$(pwd)/kpu_venv" -DPython_FIND_VIRTUALENV=FIRST
```

### Build Errors
```bash
# Clean build
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

### Import Errors
```bash
# Make sure virtual environment is activated
# Add build directory to Python path
export PYTHONPATH=build:$PYTHONPATH  # Linux/macOS
set PYTHONPATH=build;%PYTHONPATH%    # Windows
```

## Contributing

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes** in C++ and/or Python
3. **Test thoroughly**: Run both C++ and Python test suites  
4. **Update documentation** if needed
5. **Submit pull request**

## Getting Help

- **Build issues**: Check CMake output for specific errors
- **Python issues**: Verify virtual environment is activated
- **Test failures**: Run individual tests to isolate problems
- **Performance questions**: Use the visualization tools for analysis

## Development Tools

### Recommended VSCode Extensions
- **C/C++** (Microsoft)
- **Python** (Microsoft)  
- **CMake Tools** (Microsoft)
- **Jupyter** (Microsoft)

### Recommended PyCharm Plugins
- **CMake** support
- **Jupyter** support
- **C/C++** syntax highlighting

The setup scripts handle most configuration automatically, but manual setup instructions are provided for troubleshooting.
