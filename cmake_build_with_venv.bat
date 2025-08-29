# From your project directory with venv activated
kpu_dev_env\Scripts\activate

# Configure with explicit paths
cmake -S . -B build ^
    -G "Visual Studio 17 2022" ^
    -DPython_ROOT_DIR="%CD%\kpu_dev_env" ^
    -DPython_EXECUTABLE="%CD%\kpu_dev_env\Scripts\python.exe" ^
    -DPython_FIND_VIRTUALENV=FIRST ^
    -DCMAKE_BUILD_TYPE=Release

# Then open the solution in Visual Studio
start build\kpu_simulator.sln
