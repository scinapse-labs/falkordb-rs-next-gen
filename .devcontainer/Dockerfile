FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CXX=clang++-21

# Install LLVM 21 and build dependencies
RUN apt update -y && \
    apt install -y software-properties-common wget && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 21 && \
    apt update -y

# Install development dependencies including Redis
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
    lld-21 \
    redis-server \
    sudo

# Create Python virtual environment for testing
RUN python3 -m venv /data/venv

# Install Python test dependencies
RUN /data/venv/bin/pip install behave falkordb hypothesis pytest pytest-benchmark git+https://github.com/falkordb/RLTest.git@master

# Install Rust (needed before running scripts)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set working directory for build scripts
WORKDIR /workspace

# Copy build scripts
COPY graphblas.sh redisearch.sh /tmp/

# Build and install GraphBLAS using project script
RUN cd /tmp && \
    chmod +x graphblas.sh && \
    CC=clang-21 CXX=clang++-21 ./graphblas.sh

# Build and install RediSearch using project script
RUN cd /tmp && \
    chmod +x redisearch.sh && \
    ./redisearch.sh

CMD ["/bin/bash"]
