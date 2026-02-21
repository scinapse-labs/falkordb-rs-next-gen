FROM redis:8.2.0

ENV PATH="/root/.cargo/bin:${PATH}"

# Install LLVM 21 and build dependencies
RUN apt update -y && \
    apt install -y software-properties-common wget && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 21 && \
    apt update -y

# Install development dependencies
RUN apt install -y \
    git \
    make \
    cmake \
    curl \
    zlib1g \
    zlib1g-dev \
    libpolly-21-dev \
    libomp-21-dev \
    libc++-21-dev \
    libc++abi-21-dev \
    libzstd-dev \
    python3-venv \
    lcov \
    clang-21 \
    clang++-21 \
    llvm-21 \
    llvm-21-dev \
    llvm-21-tools \
    lld-21

# Create Python virtual environment for testing
RUN python3 -m venv /data/venv

# Install Python test dependencies
RUN /data/venv/bin/pip install behave falkordb hypothesis pytest pytest-benchmark git+https://github.com/falkordb/RLTest.git@master

# Build and install GraphBLAS
RUN git clone --branch v10.3.1 --single-branch https://github.com/DrTimothyAldenDavis/GraphBLAS.git && \
    cd GraphBLAS && \
    make static CMAKE_OPTIONS='-DGRAPHBLAS_COMPACT=1 -DCMAKE_POSITION_INDEPENDENT_CODE=on' JOBS=2 CC=clang-21 CXX=clang++-21 && \
    make install && \
    cd .. && \
    rm -rf GraphBLAS

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Build and install RediSearch
RUN . /data/venv/bin/activate && \
    mkdir -p /redisearch && \
    cd /redisearch && \
    git clone --recurse-submodules --branch vector-low-level-api --single-branch https://github.com/FalkorDB/RediSearch.git && \
    cd RediSearch && \
    sed -i 's/-Werror//g' deps/VectorSimilarity/src/VecSim/CMakeLists.txt && \
    make build STATIC=1 CLANG=1 CC=clang-21 CXX=clang++-21 && \
    cd .. && \
    cp -r RediSearch/bin bin  && \
    rm -rf RediSearch

# Set CXX environment variable for clippy
ENV CXX=clang++-21

# Set working directory
WORKDIR /workspace

CMD ["/bin/bash"]
