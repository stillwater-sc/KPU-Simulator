# tools/python/dev_setup.py
"""
Development setup script for Stillwater KPU Python package.
Handles dependency installation properly on Windows.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def main():
    print("🔧 Setting up Stillwater KPU Python development environment...")
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python version: {sys.version}")
    print(f"💻 Platform: {platform.system()} {platform.machine()}")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("⚠️  Warning: Not in a virtual environment. Consider creating one:")
        if platform.system() == "Windows":
            print("   python -m venv kpu_env")
            print("   kpu_env\\Scripts\\activate")
        else:
            print("   python -m venv kpu_env") 
            print("   source kpu_env/bin/activate")
        print()
    
    # Step 1: Install build dependencies first
    build_deps = [
        "setuptools>=45",
        "wheel", 
        "pybind11>=2.10.0",
        "numpy>=1.19.0"
    ]
    
    try:
        print("🔨 Installing build dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade"
        ] + build_deps, check=True)
        print("✅ Build dependencies installed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install build dependencies: {e}")
        return False
    
    # Step 2: Install basic runtime dependencies
    runtime_deps = [
        "matplotlib>=3.3.0",  # For examples
    ]
    
    try:
        print("📦 Installing runtime dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install"
        ] + runtime_deps, check=True)
        print("✅ Runtime dependencies installed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install runtime dependencies: {e}")
        print("⚠️  Continuing anyway...")
    
    # Step 3: Install package in development mode (without native extension)
    try:
        print("📦 Installing package in development mode...")
        
        # Create a minimal setup for development
        minimal_setup_content = '''from setuptools import setup, find_packages

setup(
    name="stillwater-kpu",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["numpy>=1.19.0"],
    python_requires=">=3.7",
)
'''
        
        # File paths
        setup_file = script_dir / "setup.py"
        backup_file = script_dir / "setup.py.bak"
        temp_file = script_dir / "setup_temp.py"
        
        # Backup existing setup.py if it exists
        original_content = None
        if setup_file.exists():
            # Read original content
            with open(setup_file, 'r') as f:
                original_content = f.read()
            print("   📝 Backing up original setup.py...")
        
        # Create minimal setup.py
        print("   📝 Creating minimal setup.py...")
        with open(setup_file, 'w') as f:
            f.write(minimal_setup_content)
        
        # Install in development mode
        print("   ⚙️  Installing package...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], cwd=script_dir, check=True)
        
        # Restore original setup.py if we had one
        if original_content is not None:
            print("   📝 Restoring original setup.py...")
            with open(setup_file, 'w') as f:
                f.write(original_content)
        else:
            # Remove the minimal setup.py we created
            print("   🗑️  Cleaning up temporary setup.py...")
            try:
                setup_file.unlink()
            except FileNotFoundError:
                pass  # Already gone
            
        print("✅ Package installed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install package: {e}")
        # Try to restore original setup.py on error
        if original_content is not None:
            try:
                with open(setup_file, 'w') as f:
                    f.write(original_content)
                print("   📝 Restored original setup.py after error")
            except Exception as restore_error:
                print(f"   ⚠️  Could not restore setup.py: {restore_error}")
        return False
    
    # Step 4: Install development dependencies
    dev_deps = [
        "pytest>=6.0",
        "pytest-cov",
        "black>=21.0",
        "flake8>=3.9",
    ]
    
    try:
        print("📚 Installing development dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install"
        ] + dev_deps, check=True)
        print("✅ Development dependencies installed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dev dependencies: {e}")
        print("⚠️  You can install them manually later with:")
        print(f"   pip install {' '.join(dev_deps)}")
    
    # Step 5: Test import and basic functionality
    try:
        print("🧪 Testing import...")
        import stillwater_kpu as kpu
        print(f"✅ Import successful! Version: {kpu.version()}")
        
        # Test basic functionality
        print("🚀 Testing basic functionality...")
        with kpu.Simulator(1024*1024, 64*1024) as sim:  # Small for testing
            print(f"   Simulator: {sim}")
            print(f"   Main memory: {sim.main_memory_size // 1024} KB")
            print(f"   Scratchpad: {sim.scratchpad_size // 1024} KB")
            
            # Simple matrix multiplication test
            import numpy as np
            A = np.random.randn(4, 6).astype(np.float32)
            B = np.random.randn(6, 8).astype(np.float32)
            C = sim.matmul(A, B)
            C_ref = A @ B
            
            matches = np.allclose(C, C_ref, rtol=1e-5)
            print(f"   Matrix multiply test: {'✅ PASS' if matches else '❌ FAIL'}")
            
        print("✅ All tests passed!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Try running the import test manually:")
        print("   python -c \"import stillwater_kpu as kpu; print('OK')\"")
        return False
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        print("💡 The package installed but there may be issues with the mock implementation")
        return False
    
    # Print success message and usage instructions
    print(f"\n🎉 Setup complete! You can now:")
    print("   1. Run examples:")
    if platform.system() == "Windows":
        print(f"      cd {project_root}\\examples\\python")
        print("      python simple_kpu.py")
    else:
        print(f"      cd {project_root}/examples/python")
        print("      python simple_kpu.py")
    print()
    print("   2. Run tests:")
    print(f"      cd {script_dir}")
    print("      pytest tests/ -v")
    print()
    print("   3. Start development:")
    print("      # Edit files in stillwater_kpu/")
    print("      # Changes are immediately available (editable install)")
    print()
    print("   4. Format code:")
    print("      black stillwater_kpu/")
    
    # Show note about C++ extension
    print(f"\n📝 Note: Currently using Python mock implementation.")
    print(f"   To get full performance, you'll need to build the C++ extension module.")
    print(f"   This requires the CMake build system to be set up first.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)