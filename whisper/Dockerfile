# Stage 1: Builder
FROM nvcr.io/nvidia/12.6.11-devel:12.6.11-devel-aarch64-ubuntu22.04 AS builder

ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    python3-dev \
    git \
    wget \
    cmake \
    libomp-dev \
    libopenblas-dev \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN
RUN wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb && \
    dpkg -i cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb && \
    cp /var/cudnn-local-tegra-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y --no-install-recommends libcudnn9-dev-cuda-12 && \
    rm -f cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --upgrade pip setuptools wheel pybind11 numpy pyyaml

# Build CTranslate2 with CUDA/cuDNN from the base image
RUN git clone --recursive https://github.com/OpenNMT/CTranslate2.git /tmp/CTranslate2 && \
    cd /tmp/CTranslate2 && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DWITH_CUDA=ON \
          -DWITH_CUDNN=ON \
          -DWITH_OPENBLAS=ON \
          -DWITH_MKL=OFF \
          -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
          -DOPENMP_RUNTIME=COMP \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          .. && \
    make -j4 && \
    make install && \
    cd /tmp/CTranslate2/python && \
    python3 setup.py bdist_wheel && \
    pip3 install dist/*.whl

# Build faster-whisper from source
RUN git clone -b v1.2.1 https://github.com/SYSTRAN/faster-whisper.git /tmp/faster-whisper && \
    cd /tmp/faster-whisper && \
    pip3 wheel --no-cache-dir --no-deps --wheel-dir dist .

COPY constraints.txt /tmp/constraints.txt
RUN git clone --recursive https://github.com/gpustack/vox-box.git /tmp/vox-box && \
    cd /tmp/vox-box && \
    find . -type l ! -exec test -e {} \; -delete && \
    cp /tmp/constraints.txt . && \
    pip3 wheel --no-cache-dir --no-deps --wheel-dir dist . -c /tmp/constraints.txt


# Stage 2: Final
FROM nvcr.io/nvidia/l4t-cuda:12.6.11-runtime AS final

ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    libomp-dev \
    libopenblas-dev \
    libprotobuf-dev \
    libgomp1 \
    protobuf-compiler \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN
RUN wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb && \
    dpkg -i cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb && \
    cp /var/cudnn-local-tegra-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y --no-install-recommends libcudnn9-cuda-12 && \
    rm -f cudnn-local-tegra-repo-ubuntu2204-9.3.0_1.0-1_arm64.deb && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --upgrade pip

# Copy CTranslate2 libraries and python wheel from builder
COPY --from=builder /usr/local/lib/libctranslate2.so* /usr/local/lib/
COPY --from=builder /tmp/CTranslate2/python/dist/*.whl /tmp/ctranslate2/
COPY --from=builder /tmp/faster-whisper/dist/*.whl /tmp/faster-whisper/
COPY --from=builder /tmp/vox-box/dist/*.whl /tmp/vox-box/

# Install the wheel and update library cache
RUN pip3 install --no-cache-dir /tmp/ctranslate2/*.whl && \
    pip3 install --no-cache-dir /tmp/faster-whisper/*.whl && \
    pip3 install --no-cache-dir --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 /tmp/vox-box/*.whl && \
    rm -rf /tmp/ctranslate2 /tmp/faster-whisper /tmp/vox-box && \
    ldconfig


WORKDIR /app
ENV DATA_DIR=/data
CMD ["vox-box", "start", "--huggingface-repo-id", "Systran/faster-whisper-small", "--data-dir", "/data"]