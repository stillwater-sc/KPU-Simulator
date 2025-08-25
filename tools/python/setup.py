# ============================================================================
# tools/python/setup.py
# Python package setup configuration

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import os
import sys

# Read version from CMake
def read_version():
    cmake_file = os.path.join(os.path.dirname(__file__), '..', '..', 'CMakeLists.txt')
    with open(cmake_file, 'r') as f:
        for line in f:
            if 'VERSION' in line and 'project(' in line:
                return line.split('VERSION')[1].strip().split()[0]
    return "0.0.0"

# Extension configuration
ext_modules = [
    Pybind11Extension(
        "stillwater_kpu._core",
        [
            "../../src/bindings/python/pybind_module.cpp",
        ],
        include_dirs=[
            "../../include",
            "../../src",
            "../../components/memory/include",
            "../../components/compute/include",
            "../../components/fabric/include",
            "../../components/dma/include",
            "../../components/power/include",
        ],
        libraries=["stillwater_kpu_simulator"],
        library_dirs=["../../build/lib"],
        language='c++',
        cxx_std=20,
    ),
]

setup(
    name="stillwater-kpu",
    version=read_version(),
    author="Stillwater Supercomputing",
    author_email="info@stillwater-sc.com",
    url="https://github.com/stillwater-sc/kpu-simulator",
    description="Stillwater Knowledge Processing Unit Simulator",
    long_description=open("../../README.md").read(),
    long_description_content_type="text/markdown",
    packages=["stillwater_kpu"],
    package_dir={"stillwater_kpu": "stillwater_kpu"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "viz": [
            "plotly>=5.0",
            "dash>=2.0",
            "networkx>=2.6",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
    ],
    project_urls={
        "Documentation": "https://stillwater-kpu.readthedocs.io/",
        "Source": "https://github.com/stillwater-sc/kpu-simulator",
        "Tracker": "https://github.com/stillwater-sc/kpu-simulator/issues",
    },
)

