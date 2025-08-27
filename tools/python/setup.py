from setuptools import setup, find_packages

setup(
    name="stillwater-kpu",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["numpy>=1.19.0"],
    python_requires=">=3.7",
)
