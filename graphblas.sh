#!/bin/bash
git clone --branch v10.3.1 --single-branch https://github.com/DrTimothyAldenDavis/GraphBLAS.git
cd GraphBLAS
make static CMAKE_OPTIONS='-DGRAPHBLAS_COMPACT=1 -DCMAKE_POSITION_INDEPENDENT_CODE=on' CC=clang CXX=clang++
sudo make install
cd ..
rm -rf GraphBLAS