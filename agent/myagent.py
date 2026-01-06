import logging
import os
import re
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    cli,
    metrics,
    room_io,
    llm as llm_module,
)
from livekit.plugins import openai, silero

load_dotenv()


logger = logging.getLogger("local-agent")
logger.setLevel(logging.DEBUG)


# Spanish character patterns for language detection
SPANISH_PATTERN = re.compile(r'[áéíóúñ¿¡]', re.IGNORECASE)


def detect_language(text: str) -> str:
    """
    Detect if text is in Spanish or English.
    Returns 'es' for Spanish, 'en' for English.
    """
    if not text or len(text.strip()) == 0:
        return 'en'
    
    # Check for Spanish-specific characters
    if SPANISH_PATTERN.search(text):
        return 'es'
    
    # Common Spanish words that don't have special characters
    spanish_words = {
        'hola', 'que', 'como', 'donde', 'cuando', 'quien', 'porque',
        'si', 'no', 'gracias', 'por', 'favor', 'el', 'la', 'los', 'las',
        'es', 'son', 'esta', 'estan', 'de', 'del', 'con', 'para'
    }
    
    # Tokenize and check for Spanish words
    words = text.lower().split()
    spanish_count = sum(1 for word in words if word in spanish_words)
    
    # If more than 30% of words are common Spanish words, consider it Spanish
    if len(words) > 0 and (spanish_count / len(words)) > 0.3:
        return 'es'
    
    return 'en'


# --- Agent Definitions ---
class EnglishToSpanishAgent(Agent):
    """Agent specialized in translating from English to Spanish"""
    
    def __init__(self, session_ref) -> None:
        self._session_ref = session_ref
        super().__init__(
            instructions="""
                You are an English to Spanish translator. Your ONLY job is to translate English text into Spanish.
                
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
                
                Remember: You are a one-way translator (English → Spanish only).
            """,
        )

    async def on_agent_speech_committed(self, message: llm_module.ChatMessage) -> None:
        """Set Spanish voice for output"""
        logger.info("EnglishToSpanish agent - using Spanish voice")
        
        # Use Spanish voice for output
        if self._session_ref and hasattr(self._session_ref, 'tts'):
            self._session_ref.tts._opts.voice = "ef_dora"
        
        await super().on_agent_speech_committed(message)


class SpanishToEnglishAgent(Agent):
    """Agent specialized in translating from Spanish to English"""
    
    def __init__(self, session_ref) -> None:
        self._session_ref = session_ref
        super().__init__(
            instructions="""
                You are a Spanish to English translator. Your ONLY job is to translate Spanish text into English.
                
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
                
                Remember: You are a one-way translator (Spanish → English only).
            """,
        )

    async def on_agent_speech_committed(self, message: llm_module.ChatMessage) -> None:
        """Set English voice for output"""
        logger.info("SpanishToEnglish agent - using English voice")
        
        # Use English voice for output
        if self._session_ref and hasattr(self._session_ref, 'tts'):
            self._session_ref.tts._opts.voice = "af_nicole"
        
        await super().on_agent_speech_committed(message)


# --- Server Setup ---
server = AgentServer(job_memory_warn_mb=1500)


def prewarm(proc: JobProcess):
    # Preload VAD and potentially other heavy models here
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Define plugins with your custom local URLs
    # Note: In the new pattern, we often pass these to AgentSession
    stt = openai.STT(
        base_url=os.getenv("WHISPER_URL", "http://localhost:11435/v1"),
        model="Systran/faster-whisper-small",
    )

    llm = openai.LLM(
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
        model="qwen2.5:3b",
        timeout=30,
        temperature=0.3  # Lower temperature for more consistent translations
    )

    # Use openai.TTS instead of groq.TTS
    # Default to English voice initially (will be set by agent)
    tts = openai.TTS(
        base_url=os.getenv("KOKORO_URL", "http://localhost:8880/v1"),
        model="tts-1",
        voice="af_nova",  # Default voice
        speed=1.0,
        # Streaming is automatically handled by the plugin for real-time audio
    )

    # Retrieve prewarmed VAD
    vad_inst = ctx.proc.userdata["vad"]

    # Create session first
    session = AgentSession(
        llm=llm,
        tts=tts,
        vad=vad_inst,
        stt=stt
    )

    # Language detection state
    current_agent = None
    detected_language = None

    # Custom callback to detect language and switch agents
    @session.on("user_speech_committed")
    async def on_user_speech(message: llm_module.ChatMessage):
        nonlocal current_agent, detected_language
        
        user_text = message.content
        new_language = detect_language(user_text)
        
        logger.info(f"User said: '{user_text}' | Detected language: {new_language}")
        
        # Switch agent if language changed
        if new_language != detected_language:
            detected_language = new_language
            
            if new_language == 'es':
                logger.info("Switching to SpanishToEnglish agent")
                current_agent = SpanishToEnglishAgent(session_ref=session)
            else:
                logger.info("Switching to EnglishToSpanish agent")
                current_agent = EnglishToSpanishAgent(session_ref=session)
            
            # Update the session's agent
            session._agent = current_agent

    # Start with English to Spanish agent by default
    agent_instance = EnglishToSpanishAgent(session_ref=session)
    current_agent = agent_instance
    detected_language = 'en'

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        # Log your custom metrics format here if strictly needed,
        # or use the built-in logger:
        logger.info(f"Metrics: {ev.metrics}")
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=agent_instance,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # uncomment to enable the Krisp BVC noise cancellation
                # noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user by saying: Hello, I am your English to Spanish translator.")

if __name__ == "__main__":
    cli.run_app(server)
