# How to run a python program with the stillwater_kpu python bindings

```bash
> cd <KPU_SIMULATOR_ROOT>
> PYTHONPATH="./build/Release" python ./tests/test_python.py
```

```powershell
PS> cd <KPU_SIMULATOR_ROOT>
PS> env:PYTHONPATH = ".\build\Release"
PS> $python .\tests\test_python.py

PS> Remove-Item Env:PYTHONPATH
```

# python binding

The python binding gets build

```txt
> ROOT/build/Release
```

and it doesn't have the right name:

```txt
stillwater_kpu.cp311-win_amd64.pyd
```

and the module needs to have the name
```txt
stillwater_kpu.pyd
```

You can put that file, `stillwater_kpu.pyd` somewhere in the existing PYTHONPATH, 
for example, in the scripts directory of the virtual environment, or, from the
examples above, augment the PYTHONPATH to point to the `.\build\Release` directory.
