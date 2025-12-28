#!/bin/bash
mkdir -p redisearch
cd redisearch
git clone --recurse-submodules --branch vector-low-level-api --single-branch https://github.com/FalkorDB/RediSearch.git
cd RediSearch
if [[ "$(uname -s)" == "Darwin" ]]; then
  sed -i '' 's/-Werror//g' deps/VectorSimilarity/src/VecSim/CMakeLists.txt
else
  sed -i 's/-Werror//g' deps/VectorSimilarity/src/VecSim/CMakeLists.txt
fi
make build STATIC=1 CLANG=1 CC=clang-21 CXX=clang-21 OSX_MIN_SDK_VER=15.0