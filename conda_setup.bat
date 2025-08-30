:: 1. Set up conda environment
conda create -n kpu_dev python=3.11 numpy matplotlib cmake pybind11 -c conda-forge
conda activate kpu_dev

:: 2. Navigate and clean build
cd C:\Users\tomtz\dev\stillwater\clones\KPU-simulator
rmdir /s build
mkdir build
cd build

:: 3. Configure with conda Python
cmake .. -DCMAKE_BUILD_TYPE=Debug

:: 4. Build in Visual Studio or command line
cmake --build . --config Debug