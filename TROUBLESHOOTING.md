# Troubleshooting Guide

Review this file only when the quick flow in `README.md` / `README.es.md` fails. It provides a concise checklist and the few fixes we see most often, so you can get back to the simple install steps fast.

## 1. Quick checks

Run these commands inside the environment where the build is failing:

```bash
python --version        # expect >= 3.8
cmake --version         # expect >= 3.15
git --version
```

Verify your compiler:

- Linux: `gcc --version` or `clang --version`
- macOS: `clang --version` (from Xcode Command Line Tools)
- Windows: `cl` inside “x64 Native Tools Command Prompt”, or `g++ --version` if using MinGW

Then rerun the standard flow:

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e . --no-build-isolation  # reuse local toolchain when offline
```

## 2. Common failures

| Symptom | Likely fix |
| --- | --- |
| `RuntimeError: CMake must be installed…` | Install CMake (`sudo apt-get install cmake`, `brew install cmake`, or the Windows installer) and ensure it’s on PATH. |
| `fatal error: Python.h: No such file or directory` | Install Python development headers (`python3-dev` / `python3-devel`, `xcode-select --install`). On Windows, reinstall Python and tick “Add to PATH”. |
| `FATAL_ERROR "NumPy not found…"` | `pip install numpy` in the active environment, then `python -c "import numpy"` to confirm. |
| `No CMAKE_CXX_COMPILER could be found` or MSVC errors like `C2039` | Install a C++17 compiler (build-essential, Xcode CLI tools, Visual Studio Build Tools) and regenerate with the matching CMake generator (`-G "Visual Studio 17 2022" -A x64`, etc.). |
| `CMake Error: Failed to fetch pybind11` | Check connectivity. As a fallback, clone pybind11 manually and run `cmake .. -Dpybind11_DIR=/path/to/pybind11`. |
| `ModuleNotFoundError: No module named 'wdr.coder'` after a “successful” build | Ensure you’re in the repo root, the compiled `wdr/coder.*` exists, and rerun `pip install -e .`. Remove stale artifacts if needed (see below). |
| `PermissionError` | Avoid `sudo pip` when using virtualenvs; on Windows, ensure the shell has write access to the repo folder. |

## 3. Clean rebuild

If things stay broken, wipe build artefacts and start fresh:

```bash
rm -rf build/ *.egg-info dist/ wdr/coder.cpython-*.so
pip install -e . --force-reinstall --no-cache-dir --no-build-isolation
```

Windows PowerShell / CMD:

```cmd
rmdir /s /q build
rmdir /s /q *.egg-info dist
del wdr\\coder.cpython-*.pyd
pip install -e . --force-reinstall --no-cache-dir --no-build-isolation
```
