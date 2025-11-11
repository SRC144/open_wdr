# Reproducibility Guide

This guide provides comprehensive, step-by-step instructions for building and running the WDR Compression Pipeline on **Windows**, **macOS**, and **Linux**. Follow this guide to ensure a reproducible development environment.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
   - [2.1 Unified Workflow (All Platforms)](#21-unified-workflow-all-platforms)
   - [2.2 Platform-Specific Instructions](#22-platform-specific-instructions)
3. [Build Process](#3-build-process)
4. [Verification & Testing](#4-verification--testing)
5. [Troubleshooting](#5-troubleshooting)
6. [Alternative Installation Methods](#6-alternative-installation-methods)

---

## 1. Prerequisites

### System Tools

- **Git**: Version control system (required for cloning the repository)
  - Download: https://git-scm.com/downloads
  - Verify: `git --version`

- **CMake**: Build system generator (version 3.15 or higher)
  - Download: https://cmake.org/download/
  - Verify: `cmake --version`

- **C++ Compiler**: C++17 compatible compiler
  - **Linux**: GCC 7+ or Clang 5+
  - **macOS**: Clang (included with Xcode Command Line Tools) or GCC via Homebrew
  - **Windows**: MSVC 2017+ (Visual Studio 2017 or later) or MinGW-w64

### Python

- **Python**: Version 3.8 or higher
  - Download: https://www.python.org/downloads/
  - Verify: `python --version` or `python3 --version`
  - **Important**: Python development headers are required (see platform-specific instructions)

### Python Packages (Runtime)

The following packages are automatically installed during setup, but listed here for reference:

- **numpy**: >= 1.20.0 (Numerical computing)
- **pywavelets**: >= 1.3.0 (Discrete Wavelet Transform)
- **Pillow**: >= 8.0.0 (Image I/O)

### Build-Time Dependencies

These are automatically handled by the build system:

- **pybind11**: >= 2.10.0 (Python bindings, fetched by CMake)
- **setuptools**: >= 45 (Python package builder)
- **wheel**: (Python package format)
- **setuptools-scm**: >= 6.2 (Version management)

### Testing Dependencies (Optional)

- **pytest**: >= 7.0.0 (Python testing framework)
- **pytest-cov**: >= 4.0.0 (Coverage plugin)
- **Google Test**: (C++ testing framework, fetched by CMake)

---

## 2. Environment Setup

### 2.1 Unified Workflow (All Platforms)

This is the **recommended** installation method that works across all platforms.

#### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd wdr_compression_pipeline
```

Replace `<repository-url>` with the actual repository URL.

#### Step 2: Create Python Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Step 3: Upgrade pip and Install Build Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Step 4: Install the Package

```bash
pip install -e .
```

This command will:
1. Install Python runtime dependencies (numpy, pywavelets, Pillow)
2. Build the C++ extension module using CMake
3. Compile the WDR compression core
4. Create Python bindings using pybind11
5. Install the package in editable mode

**Expected Output:**
- The `wdr_coder` module will be built as:
  - `wdr_coder.cpython-<version>-<platform>.so` (Linux/macOS)
  - `wdr_coder.cpython-<version>-<platform>.pyd` (Windows)
- The module is placed in the project root directory

#### Step 5: Verify Installation

```bash
python -c "import wdr_coder; import wdr_helpers; print('Installation successful!')"
```

If this command runs without errors, the installation was successful.

---

### 2.2 Platform-Specific Instructions

If the unified workflow fails, follow these platform-specific instructions to install system dependencies first.

#### Linux (Ubuntu/Debian)

**Install System Dependencies:**

```bash
# Update package list
sudo apt-get update

# Install build essentials (GCC, make, etc.)
sudo apt-get install -y build-essential

# Install CMake
sudo apt-get install -y cmake

# Install Python development headers
sudo apt-get install -y python3-dev python3-pip

# Install Git (if not already installed)
sudo apt-get install -y git
```

**Verify Installations:**

```bash
gcc --version          # Should show GCC 7+
cmake --version        # Should show CMake 3.15+
python3 --version      # Should show Python 3.8+
python3-config --includes  # Should show include paths
```

**Then proceed with the Unified Workflow (Section 2.1).**

#### macOS

**Option A: Using Homebrew (Recommended)**

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Xcode Command Line Tools (includes Clang)
xcode-select --install

# Install CMake
brew install cmake

# Install Python (if not using system Python)
brew install python3

# Verify installations
clang --version        # Should show Clang version
cmake --version        # Should show CMake 3.15+
python3 --version      # Should show Python 3.8+
```

**Option B: Using System Python**

If you prefer to use the system Python:

```bash
# Install Xcode Command Line Tools (required for Python headers)
xcode-select --install

# Install CMake via Homebrew
brew install cmake

# Verify Python development headers are available
python3-config --includes
```

**Then proceed with the Unified Workflow (Section 2.1).**

**Note:** On macOS, the Python module will be built as `wdr_coder.cpython-<version>-darwin.so`.

#### Windows

**Option A: Using Visual Studio Build Tools (Recommended)**

1. **Install Visual Studio Build Tools:**
   - Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
   - Run the installer and select:
     - "Desktop development with C++" workload
     - "CMake tools for Windows" (optional but recommended)
     - "Windows 10/11 SDK" (latest version)

2. **Install CMake:**
   - Download: https://cmake.org/download/
   - Select "Add CMake to system PATH" during installation
   - Or install via Visual Studio Installer (if selected above)

3. **Install Python:**
   - Download: https://www.python.org/downloads/
   - **Important**: Check "Add Python to PATH" during installation
   - **Important**: Check "Install for all users" if you want system-wide installation
   - Python development headers are included by default

4. **Install Git:**
   - Download: https://git-scm.com/download/win
   - Use default installation options

5. **Verify Installations (in Command Prompt or PowerShell):**
   ```cmd
   cl                    # Should show MSVC compiler info
   cmake --version       # Should show CMake 3.15+
   python --version      # Should show Python 3.8+
   git --version         # Should show Git version
   ```

**Option B: Using MinGW-w64**

1. **Install MinGW-w64:**
   - Download: https://www.mingw-w64.org/downloads/
   - Or use MSYS2: https://www.msys2.org/
   - Add MinGW-w64 bin directory to PATH

2. **Install CMake and Python** (same as Option A)

3. **Set Environment Variables:**
   ```cmd
   set CC=gcc
   set CXX=g++
   ```

**Then proceed with the Unified Workflow (Section 2.1).**

**Note:** On Windows, the Python module will be built as `wdr_coder.cpython-<version>-win_amd64.pyd`.

---

## 3. Build Process

### 3.1 Understanding the Build System

This project uses a **hybrid build system** that combines CMake and setuptools:

1. **CMake**: Builds the C++ code and creates the Python extension module
2. **setuptools**: Manages Python package installation and integrates with CMake

**Build Flow:**
```
setup.py (setuptools)
    ↓
CMakeBuild.build_extension()
    ↓
CMake configuration (CMakeLists.txt)
    ↓
Fetch pybind11 and googletest (via FetchContent)
    ↓
Compile C++ sources → Python extension module
    ↓
Install to Python environment
```

### 3.2 Automatic Build (Recommended)

The standard installation method automatically handles the build:

```bash
pip install -e .
```

This is equivalent to:
1. Installing Python dependencies
2. Running `setup.py` with CMake integration
3. Building the C++ extension
4. Installing the package

### 3.3 Manual Build (For Troubleshooting)

If you need to build manually for debugging:

#### Step 1: Create Build Directory

```bash
mkdir build
cd build
```

#### Step 2: Configure with CMake

**Linux/macOS:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Windows (Visual Studio):**
```cmd
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 17 2022" -A x64
```

**Windows (MinGW):**
```cmd
cmake .. -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"
```

#### Step 3: Build

**Linux/macOS:**
```bash
cmake --build . -j$(nproc)  # Use all CPU cores
# Or specify number of cores: cmake --build . -j4
```

**Windows (Visual Studio):**
```cmd
cmake --build . --config Release -j
```

**Windows (MinGW):**
```cmd
cmake --build . -j
```

#### Step 4: Locate Output

The Python extension module will be in the project root directory:
- **Linux**: `wdr_coder.cpython-<version>-linux_<arch>.so`
- **macOS**: `wdr_coder.cpython-<version>-darwin.so`
- **Windows**: `wdr_coder.cpython-<version>-win_amd64.pyd`

#### Step 5: Install Python Package

After manual build, you still need to install the Python package:

```bash
cd ..  # Return to project root
pip install -e .  # This will skip the build step if already done
```

### 3.4 Build Output Files

After a successful build, you should see:

**In project root:**
- `wdr_coder.cpython-<version>-<platform>.<ext>` (Python extension module)

**In build directory:**
- `libwdr_core.a` (Static library for C++ tests)
- Test executables (if tests were built)
- CMake cache and configuration files

### 3.5 Build Configuration Options

**CMake Variables:**
- `CMAKE_BUILD_TYPE`: `Release` (default) or `Debug`
- `CMAKE_CXX_STANDARD`: `17` (required, set automatically)
- `PYTHON_EXECUTABLE`: Automatically detected

**Compiler Flags:**
- **GCC/Clang**: `-Wall -Wextra -pedantic -O3`
- **MSVC**: `/W4 /EHsc`

---

## 4. Verification & Testing

### 4.1 Basic Verification

**Test Python Import:**
```bash
python -c "import wdr_coder; print('wdr_coder module loaded successfully')"
python -c "import wdr_helpers; print('wdr_helpers module loaded successfully')"
```

**Test Module Functions:**
```bash
python -c "import wdr_coder; help(wdr_coder.compress)"
python -c "import wdr_coder; help(wdr_coder.decompress)"
```

### 4.2 Running Python Tests

**Install Test Dependencies (if not already installed):**
```bash
pip install pytest pytest-cov
```

**Run All Python Tests:**
```bash
python -m pytest tests/
```

**Run Specific Test Files:**
```bash
python -m pytest tests/test_wdr_helpers.py
python -m pytest tests/test_wdr_coder.py
```

**Run Tests with Coverage:**
```bash
python -m pytest tests/ --cov=wdr_helpers --cov=wdr_coder
```

### 4.3 Running C++ Tests

**Build Tests:**
```bash
cd build
cmake .. -DBUILD_TESTING=ON
cmake --build .
```

**Run All C++ Tests:**
```bash
ctest
```

**Run Tests with Verbose Output:**
```bash
ctest --verbose
```

**Run Specific Test:**
```bash
./tests/test_cpp/test_arithmetic_coder  # Linux/macOS
tests\test_cpp\test_arithmetic_coder.exe  # Windows
```

### 4.4 Example Usage

**Test the Compression Pipeline:**

```bash
# Using the example script
python main.py tests_data/test_pattern.png output.wdr --reconstructed recon.png

# Verify output
ls -lh output.wdr recon.png  # Linux/macOS
dir output.wdr recon.png    # Windows
```

**Expected Output:**
- Compression statistics (file sizes, compression ratio)
- PSNR and quality metrics (if reconstruction is enabled)
- Reconstructed image file

**Python API Example:**
```python
import numpy as np
import wdr_coder
import wdr_helpers as hlp

# Load image
img = hlp.load_image("input.png")

# Perform DWT
coeffs = hlp.do_dwt(img, scales=2, wavelet='bior4.4')

# Flatten coefficients
flat_coeffs, shape_data = hlp.flatten_coeffs(coeffs)

# Compress
wdr_coder.compress(flat_coeffs, "output.wdr", num_passes=26)

# Decompress
decompressed = wdr_coder.decompress("output.wdr")

# Reconstruct
unflat_coeffs = hlp.unflatten_coeffs(decompressed, shape_data)
reconstructed = hlp.do_idwt(unflat_coeffs)

# Save
hlp.save_image("reconstructed.png", reconstructed)
```

---

## 5. Troubleshooting

### 5.1 Common Issues

#### Issue: CMake Not Found

**Symptoms:**
```
RuntimeError: CMake must be installed to build the following extensions
```

**Solutions:**
- **Linux**: `sudo apt-get install cmake`
- **macOS**: `brew install cmake` or download from https://cmake.org/download/
- **Windows**: Install from https://cmake.org/download/ and add to PATH

**Verify:** `cmake --version`

#### Issue: Python Development Headers Not Found

**Symptoms:**
```
fatal error: Python.h: No such file or directory
```

**Solutions:**
- **Linux (Ubuntu/Debian)**: `sudo apt-get install python3-dev`
- **Linux (Fedora/RHEL)**: `sudo dnf install python3-devel`
- **macOS**: Install Xcode Command Line Tools: `xcode-select --install`
- **Windows**: Python headers are included by default with Python installation

**Verify:**
```bash
python3-config --includes  # Linux/macOS
# Should show include paths like: -I/usr/include/python3.8
```

#### Issue: NumPy Not Found During Build

**Symptoms:**
```
FATAL_ERROR "NumPy not found. Please install it with: pip install numpy"
```

**Solutions:**
1. Ensure NumPy is installed in the active Python environment:
   ```bash
   pip install numpy
   ```

2. Verify NumPy installation:
   ```bash
   python -c "import numpy; print(numpy.__version__)"
   ```

3. If using a virtual environment, ensure it's activated before building.

#### Issue: Compiler Not Found or Wrong Version

**Symptoms:**
```
No CMAKE_CXX_COMPILER could be found
```

**Solutions:**
- **Linux**: Install build-essential: `sudo apt-get install build-essential`
- **macOS**: Install Xcode Command Line Tools: `xcode-select --install`
- **Windows**: Install Visual Studio Build Tools with C++ workload

**Verify Compiler:**
```bash
gcc --version    # Linux/macOS (GCC)
clang --version  # macOS/Linux (Clang)
cl              # Windows (MSVC)
```

#### Issue: pybind11 Fetch Fails

**Symptoms:**
```
CMake Error: Failed to fetch pybind11
```

**Solutions:**
1. Check internet connection (CMake fetches pybind11 from GitHub)
2. If behind a proxy, configure CMake proxy settings
3. Manually download pybind11 and set `pybind11_DIR`:
   ```bash
   git clone https://github.com/pybind/pybind11.git
   cmake .. -Dpybind11_DIR=/path/to/pybind11
   ```

#### Issue: Module Import Error After Build

**Symptoms:**
```
ModuleNotFoundError: No module named 'wdr_coder'
```

**Solutions:**
1. Ensure the module file exists in the project root:
   ```bash
   ls wdr_coder.cpython-*.so  # Linux/macOS
   dir wdr_coder.cpython-*.pyd  # Windows
   ```

2. Ensure you're in the project root directory when importing

3. Reinstall the package:
   ```bash
   pip uninstall wdr-compression-pipeline
   pip install -e .
   ```

4. Check Python path:
   ```python
   import sys
   print(sys.path)  # Should include project root
   ```

#### Issue: Permission Denied Errors

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
- **Linux/macOS**: Avoid using `sudo` with pip in virtual environments
- **Windows**: Run Command Prompt/PowerShell as Administrator if needed
- Ensure write permissions to the project directory

#### Issue: Build Fails on Windows with MSVC

**Symptoms:**
```
error C2039: 'string': is not a member of 'std'
```

**Solutions:**
1. Ensure you're using Visual Studio 2017 or later
2. Use the correct CMake generator:
   ```cmd
   cmake .. -G "Visual Studio 17 2022" -A x64
   ```
3. Ensure Windows SDK is installed via Visual Studio Installer

### 5.2 Platform-Specific Issues

#### Linux: Multiple Python Versions

If you have multiple Python versions installed:

```bash
# Use python3 explicitly
python3 -m venv .venv
python3 -m pip install -e .

# Or specify Python version for CMake
cmake .. -DPYTHON_EXECUTABLE=/usr/bin/python3.8
```

#### macOS: Architecture Mismatch (Apple Silicon)

If building on Apple Silicon (M1/M2) Macs:

```bash
# Ensure you're using the correct architecture
arch -arm64 python -m pip install -e .  # For Apple Silicon
arch -x86_64 python -m pip install -e .  # For Intel (via Rosetta)
```

**Note:** The build system should automatically detect the architecture.

#### Windows: Path Issues

If Python or CMake are not found:

1. **Add to PATH manually:**
   - Python: `C:\Python38\` and `C:\Python38\Scripts\`
   - CMake: `C:\Program Files\CMake\bin\`

2. **Use full paths:**
   ```cmd
   "C:\Program Files\CMake\bin\cmake.exe" --version
   "C:\Python38\python.exe" --version
   ```

3. **Restart terminal** after modifying PATH

### 5.3 Clean Build

If you encounter persistent build issues, try a clean build:

```bash
# Remove build artifacts
rm -rf build/              # Linux/macOS
rmdir /s /q build          # Windows

# Remove Python build artifacts
rm -rf *.egg-info dist/    # Linux/macOS
rmdir /s /q *.egg-info dist  # Windows

# Remove compiled modules
rm -f wdr_coder.cpython-*.so wdr_coder.cpython-*.pyd

# Rebuild
pip install -e . --force-reinstall --no-cache-dir
```

---

## 6. Alternative Installation Methods

### 6.1 Conda/Mamba Environment

Conda/Mamba can manage both Python and C++ dependencies, providing better reproducibility.

#### Create Environment File

Create `environment.yml`:

```yaml
name: wdr-compression
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.8
  - numpy>=1.20.0
  - pywavelets>=1.3.0
  - pillow>=8.0.0
  - cmake>=3.15
  - compilers  # C++ compiler
  - pip
  - pip:
    - pybind11>=2.10.0
```

#### Install with Conda

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate wdr-compression

# Install the package
pip install -e .
```

**Note:** Conda provides pre-compiled packages, which may speed up installation but the C++ extension still needs to be built.

### 6.2 Docker (Optional)

For maximum reproducibility, you can use Docker:

#### Create Dockerfile

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Install the package
RUN pip install -e .

# Default command
CMD ["python", "main.py", "--help"]
```

#### Build and Run

```bash
# Build image
docker build -t wdr-compression .

# Run container
docker run -v $(pwd):/data wdr-compression python main.py /data/input.png /data/output.wdr
```

### 6.3 System Package Installation

For system-wide installation (not recommended for development):

```bash
# Install without editable mode
pip install .

# Or with specific options
pip install . --user  # Install to user directory
```

**Note:** Editable installation (`pip install -e .`) is recommended for development as it allows code changes without reinstalling.

---

## 7. Additional Resources

### Project Documentation

- **README.md**: Main project documentation
- **README.es.md**: Spanish documentation
- **docs/theory.md**: Theoretical explanation of the WDR algorithm
- **docs/theory.es.md**: Theoretical explanation (Spanish)

### External Resources

- **CMake Documentation**: https://cmake.org/documentation/
- **pybind11 Documentation**: https://pybind11.readthedocs.io/
- **Python Packaging Guide**: https://packaging.python.org/
- **NumPy Documentation**: https://numpy.org/doc/
- **PyWavelets Documentation**: https://pywavelets.readthedocs.io/

### Getting Help

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting](#5-troubleshooting) section
2. Review project issues on the repository
3. Consult the main [README.md](README.md) for additional information
4. Verify all prerequisites are correctly installed

---

## 8. Quick Reference

### Installation Commands (All Platforms)

```bash
# Clone repository
git clone <repository-url>
cd wdr_compression_pipeline

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate      # Windows

# Install
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

# Verify
python -c "import wdr_coder; import wdr_helpers; print('Success!')"
```

### Build Commands (Manual)

```bash
# Configure
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j

# Return to root
cd ..
```

### Test Commands

```bash
# Python tests
python -m pytest tests/

# C++ tests
cd build && ctest
```

---

**Last Updated**: This guide is maintained alongside the project. For the latest version, see the repository.

