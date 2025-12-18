FROM nvcr.io/nvidia/l4t-cuda:12.6.11-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    python3-dev \
    git \
    cmake \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --upgrade pip 

# Build CTranslate2 with CUDA support from source
RUN git clone --recursive https://github.com/OpenNMT/CTranslate2.git /tmp/CTranslate2 && \
    cd /tmp/CTranslate2 && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DWITH_CUDA=ON \
          -DWITH_CUDNN=ON \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          .. && \
    make -j$(nproc) && \
    make install && \
    cd /tmp/CTranslate2/python && \
    pip3 install --no-cache-dir . && \
    rm -rf /tmp/CTranslate2

# Then install vox-box
RUN pip3 install --no-cache-dir --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 vox-box 

WORKDIR /app

ENV DATA_DIR=/data

CMD ["vox-box", "start", "--huggingface-repo-id", "Systran/faster-whisper-small", "--data-dir", "/data"]