# PyTorch JIT in C++

This folder contains the code to load a TorchScript model into
the C++ runtime of PyTorch and profile it.

## Installation
To build the example application you will need:
* libtorch: Unpack the zip file in this directory. Get the CPU-only version
        [here](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.2.0.zip),
        as it has fewer dependencies.
* C++ compiler: Either the g++ or Visual Studio 15 2017 compiler. More info
        [here](https://github.com/pytorch/pytorch#from-source).
* CMake: I used 3.13.4 on Windows.

Build the application with:
```
mkdir build
# For Linux
    cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch -B build
# For Windows
    cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch -B build -G "Visual Studio 15 2017 Win64"
cd build
# For Linux
    make
# For Windows
    msbuild 
```

## Usage
Use the application with any model from torchvision:
```
# For Linux
./build/Debug/example_app /path/to/traced/model.pth
# For Windows
./build/Debug/example_app.exe /path/to/traced/model.pth
```
