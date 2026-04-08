#!/bin/bash

# Step 1: Check if we need to download and patch
if [ ! -d "Pangolin" ]; then
    echo "Pangolin not found. Cloning and patching..."
    git clone -b v0.6 https://github.com/stevenlovegrove/Pangolin.git
    cd Pangolin
    sed -i '1i#include <limits>' include/pangolin/gl/colour.h
    cd ..
else
    echo "Pangolin directory already exists. Proceeding directly to build..."
fi

# Step 2: Always build and install
cd Pangolin
mkdir -p build  # -p ensures it doesn't throw an error if the folder exists
cd build

# Run CMake and Make
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PANGOLIN_VIDEO=OFF -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j$(nproc)
make install

# Return to the root directory
cd ../..