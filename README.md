# Voice AI Agent - Bilingual Translation

Real-time English ↔ Spanish translation agent powered by Pipecat framework with local AI models.

## 🎯 Features

- **Automatic Language Detection**: Detects whether you're speaking English or Spanish
- **Real-time Translation**: Translates speech to the other language in real-time
- **Bilingual Voice**: Uses appropriate voice for each language
  - Spanish output: ef_dora voice
  - English output: af_nicole voice
- **Local AI Models**: Runs entirely on local hardware (no cloud API costs)
- **Modern UI**: Built with Pipecat Voice UI Kit and Next.js

## 🏗️ Architecture

**Framework**: [Pipecat](https://github.com/pipecat-ai/pipecat) - Python framework for voice AI agents

**AI Services** (all local):
- **STT**: faster-whisper-small (Systran) via local Whisper server
- **LLM**: qwen2.5:3b via local Ollama
- **TTS**: Kokoro TTS (multilingual)
- **VAD**: Silero Voice Activity Detection

**Transport**: SmallWebRTC (development mode)

## 🚀 Quick Start

```bash
# Start all services
./start.sh

# Or manually with Docker Compose
docker-compose up --build

# Access the web interface
open http://localhost:3000
```

## 📋 Requirements

- Docker & Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- ~8GB GPU VRAM
- ~10GB disk space for models

## 🔧 Services & Ports

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 3000 | Next.js web interface |
| Pipecat Agent | 7860 | Voice AI agent (WebRTC) |
| Kokoro TTS | 8880 | Text-to-speech synthesis |
| Whisper STT | 11435 | Speech-to-text transcription |
| Ollama LLM | 11434 | Local language model |

## 💡 How It Works

1. User speaks in English or Spanish
2. Whisper transcribes speech to text
3. Language detector identifies the language
4. System switches translation direction:
   - English → Spanish (with Spanish voice)
   - Spanish → English (with English voice)
5. Ollama translates the text
6. Kokoro synthesizes speech in target language
7. User hears the translation

## 📁 Project Structure

```
voice-ai-agent/
├── agent/                      # Pipecat agent
│   ├── translator_agent.py    # Main agent logic
│   ├── services/               # Custom service wrappers
│   │   ├── local_whisper_stt.py
│   │   ├── local_ollama_llm.py
│   │   └── local_kokoro_tts.py
│   ├── requirements.txt        # Python dependencies
│   └── Dockerfile
├── frontend/                   # Next.js + Tailwind frontend
│   ├── src/
│   │   └── app/
│   │       ├── api/offer/      # WebRTC offer endpoint
│   │       ├── components/     # React components
│   │       └── page.tsx        # Main page with Voice UI Kit
│   ├── Dockerfile
│   └── package.json
├── kokoro/                     # Kokoro TTS service
├── whisper/                    # Whisper STT service
├── ollama/                     # Ollama LLM service
├── docker-compose.yml          # Service orchestration
└── start.sh                    # Quick start script
```

## 🎨 Frontend

The frontend is built with:
- **Next.js 15** with App Router
- **Tailwind CSS 4** for styling
- **Pipecat Voice UI Kit** for voice components
- **Three.js** for WebGL visualizations

Based on the [Pipecat Voice UI Kit Tailwind example](https://github.com/pipecat-ai/voice-ui-kit/tree/main/examples/03-tailwind).

See [frontend/FRONTEND_DOCS.md](frontend/FRONTEND_DOCS.md) for detailed documentation.

## 🔄 Migration from LiveKit

This project was migrated from LiveKit to Pipecat. See [PIPECAT_MIGRATION.md](PIPECAT_MIGRATION.md) for details.

**Key Benefits**:
- Simpler architecture (no separate media server)
- Easier local development
- More flexible AI service integration
- Purpose-built for AI agents

## 🛠️ Development

### Running Individual Services

```bash
# Run only the agent
docker-compose up agent

# Run without frontend
docker-compose up kokoro whisper ollama agent

# View logs
docker-compose logs -f agent
```

### Environment Variables

Edit `.env` or set in `docker-compose.yml`:

```bash
# Agent
WHISPER_URL=http://localhost:11435/v1
OLLAMA_URL=http://localhost:11434/v1
KOKORO_URL=http://localhost:8880/v1

# Frontend
AGENT_URL=http://localhost:7860/api/offer
```

## 🎨 Customization

### Change Translation Languages

Edit [agent/translator_agent.py](agent/translator_agent.py#L60):
- Update `detect_language()` function
- Modify system prompts for different language pairs
- Change TTS voices in voice mapping

### Change LLM Model

Edit [docker-compose.yml](docker-compose.yml) or pull a different model:

```bash
docker exec -it $(docker ps -q -f name=ollama) ollama pull llama3
```

Then update `model="llama3"` in [agent/services/local_ollama_llm.py](agent/services/local_ollama_llm.py).

### Change TTS Voices

Available Kokoro voices:
- English: af_nova, af_nicole, af_sarah, am_adam, am_michael
- Spanish: ef_dora, ef_bella, em_diego

Edit voice mapping in [agent/translator_agent.py](agent/translator_agent.py).

## 📊 Monitoring

The Voice UI Kit includes a built-in debug panel showing:
- Connection status
- Audio levels
- Transcriptions (real-time)
- Agent responses
- Latency metrics

## 🐛 Troubleshooting

**Agent not connecting:**
- Check if all services are running: `docker-compose ps`
- Verify ports are not in use: `netstat -tuln | grep -E '(3000|7860|8880|11434|11435)'`

**Poor translation quality:**
- Try a larger Ollama model (e.g., qwen2.5:7b)
- Adjust LLM temperature in [local_ollama_llm.py](agent/services/local_ollama_llm.py)

**Slow response time:**
- Check GPU usage: `nvidia-smi`
- Reduce model sizes or use GPU acceleration
- Ensure NVIDIA runtime is being used

## 📝 License

See individual service licenses:
- Pipecat: Apache 2.0
- Whisper: MIT
- Ollama: MIT
- Kokoro TTS: Check dustynv/kokoro-tts license

## 🙏 Credits

Built with:
- [Pipecat](https://github.com/pipecat-ai/pipecat) by Pipecat AI
- [Ollama](https://github.com/ollama/ollama) by Ollama
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN
- [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) by various contributors
