#!/bin/bash
set -e

LAGRAPH_VERSION="v1.2.1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR/lagraph_lib"

if [ -d "LAGraph" ]; then
    rm -rf LAGraph
fi
git clone --branch "$LAGRAPH_VERSION" --single-branch https://github.com/GraphBLAS/LAGraph.git
cd LAGraph

mkdir -p build
cd build

# Set OpenMP flags for macOS with Homebrew LLVM
if [ "$(uname)" = "Darwin" ]; then
    export CC=/opt/homebrew/opt/llvm/bin/clang
    export CXX=/opt/homebrew/opt/llvm/bin/clang++
    OPENMP_FLAGS="-DOpenMP_C_FLAGS=-fopenmp -DOpenMP_C_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/llvm/lib/libomp.dylib -DOpenMP_CXX_FLAGS=-fopenmp -DOpenMP_CXX_LIB_NAMES=omp"
else
    OPENMP_FLAGS=""
fi

cmake .. \
    -DBUILD_STATIC_LIBS=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLIBRARY_ONLY=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DGRAPHBLAS_INCLUDE_DIR=/usr/local/include/suitesparse \
    -DGRAPHBLAS_LIBRARY=/usr/local/lib/libgraphblas.a \
    -DSUITESPARSE_USE_FORTRAN=OFF \
    $OPENMP_FLAGS

make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# Install to project directory
mkdir -p "$INSTALL_DIR"
cp src/liblagraph.a "$INSTALL_DIR/"
cp experimental/liblagraphx.a "$INSTALL_DIR/"
cp ../include/LAGraph.h "$INSTALL_DIR/"
cp ../include/LAGraphX.h "$INSTALL_DIR/"

cd "$SCRIPT_DIR"
rm -rf LAGraph
