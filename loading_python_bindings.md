\# Loading the python bindings



The absolute path version doesn't work on a bash emulator on Windows:



```bash

cd ~/dev/stillwater/clones/KPU-simulator

PYTHONPATH="C:/Users/tomtz/dev/stillwater/clones/KPU-simulator/build/Release:$PYTHONPATH" python -c "import sys; print('\\\\n'.join(sys.path))"





C:\\Users\\tomtz\\dev\\stillwater\\clones\\KPU-simulator\\build\\Release:

C:\\Python\\Python311\\python311.zip

C:\\Python\\Python311\\DLLs

C:\\Python\\Python311\\Lib

C:\\Python\\Python311

C:\\Users\\tomtz\\dev\\stillwater\\clones\\KPU-simulator\\kpu\_venv

C:\\Users\\tomtz\\dev\\stillwater\\clones\\KPU-simulator\\kpu\_venv\\Lib\\site-packages

C:\\Users\\tomtz\\dev\\stillwater\\clones\\KPU-simulator\\kpu\_venv\\Lib\\site-packages\\win32

C:\\Users\\tomtz\\dev\\stillwater\\clones\\KPU-simulator\\kpu\_venv\\Lib\\site-packages\\win32\\lib

C:\\Users\\tomtz\\dev\\stillwater\\clones\\KPU-simulator\\kpu\_venv\\Lib\\site-packages\\Pythonwin

```



A weird tailing `:` for the path that needs to work. Using PYTHONPATH="./build/Release" with the relative path works correctly.



```bash

tomtz@A8 MINGW64 ~/dev/stillwater/clones/KPU-simulator (main)

$ PYTHONPATH="./build/Release" python ./tests/test\_python.py

✓ Successfully imported stillwater\_kpu vVERSION\_INFO

KPU Simulator Python Bindings Test Suite

...

```



The problem is bash on Windows is having issues with the absolute path format. The relative path ./build/Release is properly resolved by Python's import system.



Thus, a working solution for running tests with the compiled C++ module is:

```bash

cd ~/dev/stillwater/clones/KPU-simulator

PYTHONPATH="./build/Release" python ./tests/test\_python.py

```



For a more permanent solution, you could add this to your shell profile or create a simple script. 



You can also copy the .pyd file to your virtual environment's site-packages directory:

```bash

cp "./build/Release/stillwater\_kpu.pyd" "./kpu\_venv/Lib/site-packages/"

```



Then you wouldn't need to set PYTHONPATH at all - the module would be available in any Python script run with that virtual environment active.

