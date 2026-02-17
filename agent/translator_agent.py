"""
Bilingual Translation Agent using Pipecat Framework.
Automatically detects language and translates English ↔ Spanish.
"""

import os
import re
from typing import Dict, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

# Import custom local service wrappers
from services.local_whisper_stt import LocalWhisperSTTService
from services.local_ollama_llm import LocalOllamaLLMService
from services.local_kokoro_tts import LocalKokoroTTSService

load_dotenv(override=True)

# FastAPI app for local development with frontend compatibility
app = FastAPI()

# Store connections by pc_id for frontend compatibility
pcs_map: Dict[str, SmallWebRTCConnection] = {}

# ICE servers configuration
ice_servers = [
    IceServer(urls="stun:stun.l.google.com:19302")
]


# Spanish character pattern for language detection
SPANISH_PATTERN = re.compile(r"[áéíóúñ¿¡]", re.IGNORECASE)


def detect_language(text: str) -> str:
    """
    Detect if text is in Spanish or English.
    Returns 'es' for Spanish, 'en' for English.
    """
    if not text or len(text.strip()) == 0:
        return "en"

    # Check for Spanish-specific characters
    if SPANISH_PATTERN.search(text):
        return "es"

    # Common Spanish words that don't have special characters
    spanish_words = {
        "hola",
        "que",
        "como",
        "donde",
        "cuando",
        "quien",
        "porque",
        "si",
        "no",
        "gracias",
        "por",
        "favor",
        "el",
        "la",
        "los",
        "las",
        "es",
        "son",
        "esta",
        "estan",
        "de",
        "del",
        "con",
        "para",
    }

    # Tokenize and check for Spanish words
    words = text.lower().split()
    spanish_count = sum(1 for word in words if word in spanish_words)

    # If more than 30% of words are common Spanish words, consider it Spanish
    if len(words) > 0 and (spanish_count / len(words)) > 0.3:
        return "es"

    return "en"


class TextFilterProcessor(FrameProcessor):
    """
    Processor that filters out empty or whitespace-only text frames.
    Prevents TTS errors from receiving empty strings.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Only filter TextFrames
        if isinstance(frame, TextFrame):
            text = frame.text
            # Skip empty or whitespace-only text
            if not text or not text.strip():
                logger.debug(f"Filtering empty text frame: '{text}'")
                return
        
        # Pass all valid frames through (super() handles pushing)
        await super().process_frame(frame, direction)


class LanguageDetectionProcessor(FrameProcessor):
    """
    Processor that detects language from user transcriptions
    and updates LLM context + TTS voice accordingly.
    """

    def __init__(self, llm_service, tts_service, context):
        super().__init__()
        self._llm_service = llm_service
        self._tts_service = tts_service
        self._context = context
        self._current_language: Optional[str] = None
        
        # Translation instructions for each direction
        self._english_to_spanish_instruction = """You are an English to Spanish translator. Your ONLY job is to translate English text into Spanish.

Translation rules:
- You receive input in ENGLISH
- You output the SPANISH translation
- Translate word-for-word, preserving meaning and tone

CRITICAL - What you must NEVER do:
- Do NOT respond to questions or engage in conversation
- Do NOT answer what the user is asking
- Do NOT provide explanations or comments
- Do NOT translate from Spanish to English (wrong direction!)
- ONLY translate English → Spanish

Examples:
Input: "How are you today?"
Output: "¿Cómo estás hoy?"

Input: "I like pizza"
Output: "Me gusta la pizza"

Input: "Where is the bathroom?"
Output: "¿Dónde está el baño?"

Remember: You are a one-way translator (English → Spanish only)."""

        self._spanish_to_english_instruction = """You are a Spanish to English translator. Your ONLY job is to translate Spanish text into English.

Translation rules:
- You receive input in SPANISH
- You output the ENGLISH translation
- Translate word-for-word, preserving meaning and tone

CRITICAL - What you must NEVER do:
- Do NOT respond to questions or engage in conversation
- Do NOT answer what the user is asking
- Do NOT provide explanations or comments
- Do NOT translate from English to Spanish (wrong direction!)
- ONLY translate Spanish → English

Examples:
Input: "¿Cómo estás hoy?"
Output: "How are you today?"

Input: "Me gusta la pizza"
Output: "I like pizza"

Input: "¿Dónde está el baño?"
Output: "Where is the bathroom?"

Remember: You are a one-way translator (Spanish → English only)."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Detect language from user transcriptions
        if isinstance(frame, TranscriptionFrame):
            text = frame.text
            detected_lang = detect_language(text)

            logger.info(f"User said: '{text}' | Detected language: {detected_lang}")

            # Switch translation direction if language changed
            if detected_lang != self._current_language:
                self._current_language = detected_lang
                await self._switch_language_mode(detected_lang)

        await self.push_frame(frame, direction)

    async def _switch_language_mode(self, language: str):
        """Switch translation mode and TTS voice based on detected language."""
        if language == "es":
            # Spanish input → English output
            logger.info("Switching to Spanish→English translation mode")
            system_instruction = self._spanish_to_english_instruction
            tts_voice = "af_nicole"  # English voice for output
        else:
            # English input → Spanish output
            logger.info("Switching to English→Spanish translation mode")
            system_instruction = self._english_to_spanish_instruction
            tts_voice = "ef_dora"  # Spanish voice for output

        # Update LLM system message
        messages = self._context.messages.copy()
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_instruction
        else:
            messages.insert(0, {"role": "system", "content": system_instruction})
        
        self._context.set_messages(messages)

        # Update TTS voice
        self._tts_service.set_voice(tts_voice)
        logger.info(f"✓ Updated TTS voice to: {tts_voice}")


async def run_bot(transport: BaseTransport, runner_args: Optional[RunnerArguments] = None):
    """Run the translation bot with the provided transport.

    Args:
        transport (BaseTransport): The transport to use for communication.
        runner_args: runner session arguments (optional for local mode)
    """
    if runner_args:
        logger.info(f"RunnerArguments custom data: {runner_args.body}")
    else:
        logger.info("Running in local development mode")
    
    # Initialize AI services with local endpoints
    stt = LocalWhisperSTTService(
        base_url=os.getenv("WHISPER_URL", "http://localhost:11435/v1"),
        model="Systran/faster-whisper-small",
    )

    llm = LocalOllamaLLMService(
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
        model="gemma3:1b",
    )

    tts = LocalKokoroTTSService(
        base_url=os.getenv("KOKORO_URL", "http://localhost:8880/v1"),
        model="tts-1",
        voice="af_nova",  # Default English voice
    )

    # Create LLM context with initial system message (English→Spanish)
    messages = [
        {
            "role": "system",
            "content": "You are an English to Spanish translator. Translate the user's English text to Spanish.",
        }
    ]
    
    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # Create language detection processor
    language_detector = LanguageDetectionProcessor(llm, tts, context)

    # Create text filter to prevent empty strings from reaching TTS
    text_filter = TextFilterProcessor()

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Build pipeline: order matters!
    pipeline = Pipeline(
        [
            transport.input(),  # Receive audio from user
            rtvi,  # RTVI processor for client events
            stt,  # Speech → Text (transcription)
            language_detector,  # Detect language & switch mode
            context_aggregator.user(),  # Add user message to LLM context
            llm,  # Generate translation
            text_filter,  # Filter empty text frames
            tts,  # Text → Speech
            transport.output(),  # Send audio to user
            context_aggregator.assistant(),  # Save bot response to context
        ]
    )

    # Create pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    # Event handlers
    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.debug("Client ready event received")
        await rtvi.set_bot_ready()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected.")
        # Kick off the conversation
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, participant):
        logger.info("Client disconnected: {}", participant)
        await task.cancel()

    # Run the pipeline
    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint if runner_args else False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    logger.info(f"Starting the bot, received body: {runner_args.body}")
    webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection
    try:
        if os.environ.get("ENV") != "local":
            from pipecat.audio.filters.krisp_filter import KrispFilter

            krisp_filter = KrispFilter()
        else:
            krisp_filter = None

        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_in_filter=krisp_filter,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        if transport is None:
            logger.error("Failed to create transport")
            return

        await run_bot(transport, runner_args)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise


# FastAPI endpoints for frontend compatibility (local development)
@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    """Handle WebRTC offer from frontend - local development mode."""
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"WebRTC connection closed: {webrtc_connection.pc_id}")
            if webrtc_connection.pc_id in pcs_map:
                del pcs_map[webrtc_connection.pc_id]

        # Create transport directly for local mode
        transport = SmallWebRTCTransport(
            webrtc_connection=pipecat_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        # Run bot in background without RunnerArguments (local mode)
        background_tasks.add_task(run_bot, transport, None)

    answer = pipecat_connection.get_answer()
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@app.patch("/api/offer")
async def patch_offer(request: dict):
    """Handle ICE candidate updates from frontend."""
    pc_id = request.get("pc_id")
    
    if not pc_id or pc_id not in pcs_map:
        return {"error": "Connection not found"}
    
    pipecat_connection = pcs_map[pc_id]
    
    # Handle ICE candidates
    if "candidate" in request:
        candidate = request["candidate"]
        await pipecat_connection.add_ice_candidate(candidate)
    
    return {"status": "ok"}


if __name__ == "__main__":
    # Check if running in Pipecat Cloud mode or local development mode
    use_cloud_runner = os.getenv("PIPECAT_CLOUD", "false").lower() == "true"
    
    if use_cloud_runner:
        # Use Pipecat Cloud runner
        from pipecat.runner.run import main
        
        logger.info("🤖 Starting in Pipecat Cloud mode")
        main()
    else:
        # Use local FastAPI server for frontend compatibility
        logger.info("🤖 Starting Bilingual Translation Agent (Local Development)")
        logger.info("🌐 WebRTC transport available on port 7860")
        logger.info("📝 Speak in English or Spanish - I'll translate to the other language!")
        
        uvicorn.run(app, host="0.0.0.0", port=7860)
