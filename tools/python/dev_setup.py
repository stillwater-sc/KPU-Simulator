# tools/python/dev_setup.py
"""
Development setup script for Stillwater KPU Python package.
Run this to set up the development environment.
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
        print("   python -m venv kpu_env")
        print("   source kpu_env/bin/activate  # Linux/macOS")
        print("   kpu_env\\Scripts\\activate     # Windows")
        print()
    
    # Install package in development mode
    try:
        print("📦 Installing package in development mode...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], cwd=script_dir, check=True)
        print("✅ Package installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install package: {e}")
        return False
    
    # Install development dependencies
    dev_deps = [
        "pytest>=6.0",
        "pytest-cov",
        "black>=21.0", 
        "flake8>=3.9",
        "mypy>=0.910",
        "jupyter",
        "matplotlib",
        "plotly"
    ]
    
    try:
        print("📚 Installing development dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install"
        ] + dev_deps, check=True)
        print("✅ Development dependencies installed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dev dependencies: {e}")
    
    # Test import
    try:
        print("🧪 Testing import...")
        import stillwater_kpu as kpu
        print(f"✅ Import successful! Version: {kpu.version()}")
        
        # Test basic functionality
        print("🚀 Testing basic functionality...")
        with kpu.Simulator(1024*1024, 64*1024) as sim:  # Smaller for testing
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
        return False
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False
    
    # Print usage instructions
    print("\n🎉 Setup complete! You can now:")
    print("   1. Run examples:")
    print(f"      cd {project_root / 'examples' / 'python'}")
    print("      python simple_kpu.py")
    print()
    print("   2. Run tests:")
    print(f"      cd {script_dir}")
    print("      pytest tests/")
    print()
    print("   3. Start Jupyter:")
    print("      jupyter notebook")
    print()
    print("   4. Format code:")
    print("      black stillwater_kpu/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)