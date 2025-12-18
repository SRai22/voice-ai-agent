FROM nvcr.io/nvidia/l4t-cuda:12.6.11-runtime

# Install Python and other dependencies
RUN apt-get update -y &&  \
    apt-get install -y python3.10 python3.10-dev python3-venv espeak-ng espeak-ng-data git libsndfile1 curl ffmpeg g++ cmake libmecab-dev mecab mecab-ipadic-utf8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/share/espeak-ng-data &&  \
    ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/ && \
    useradd -m -u 1001 appuser && \
    mkdir -p /app/api/src/models/v1_0 && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml ./pyproject.toml

# Install dependencies with GPU extras
# Pre-install PyTorch from Jetson AI Lab for ARM64/CUDA 12.6 compatibility
RUN uv venv --python 3.10 && \
    uv pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch && \
    uv sync --extra gpu --no-cache --no-install-project --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --index-strategy unsafe-best-match

# Copy project files including models
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser docker/scripts/ ./
RUN chmod +x ./entrypoint.sh


# Set all environment variables in one go
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    DEVICE="gpu"

ENV DOWNLOAD_MODEL=true
# Download model if enabled
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
    python download_model.py --output api/src/models/v1_0; \
    fi

# Run FastAPI server through entrypoint.sh
CMD ["./entrypoint.sh"]
