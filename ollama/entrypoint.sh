#!/bin/sh
set -e

echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama server to be ready..."
# Wait for ollama to be ready by checking the API
READY=0
for i in $(seq 1 60); do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama server is ready!"
        READY=1
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 2
done

if [ $READY -eq 0 ]; then
    echo "ERROR: Ollama server failed to start within 120 seconds"
    exit 1
fi

# Check if model exists, if not pull it
if ! curl -s http://localhost:11434/api/tags | grep -q "qwen2.5:3b"; then
    echo "Downloading qwen2.5:3b model..."
    if ! ollama pull qwen2.5:3b; then
        echo "ERROR: Failed to download qwen2.5:3b model"
        exit 1
    fi
    echo "Model downloaded successfully!"
fi

echo "Setup complete, keeping container running..."
wait $OLLAMA_PID