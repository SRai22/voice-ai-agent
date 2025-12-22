import logging
import os
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
)
from livekit.plugins import openai, silero, groq

load_dotenv()

logger = logging.getLogger("local-agent")
logger.setLevel(logging.INFO)

# --- Agent Definition ---
class LocalAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a helpful agent.
                Never ever use emojis. Everything you say should be in plain text, since it will be spoken out loud.
                Keep your responses short and concise. Never more than a sentence or two.
            """,
        )


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
        model="Systran/faster-whisper-small"
    )
    
    llm = openai.LLM(
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), 
        model="gemma3:1b", 
        timeout=30
    )
    tts = groq.TTS(
        base_url=os.getenv("KOKORO_URL", "http://localhost:8880/v1"), 
        model="kokoro", 
        voice="af_nova"
    )
    
    # Retrieve prewarmed VAD
    vad_inst = ctx.proc.userdata["vad"]

    session = AgentSession(
        llm=llm,
        tts=tts,
        vad=vad_inst,
        stt=stt
    )

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
        agent=LocalAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # uncomment to enable the Krisp BVC noise cancellation
                # noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user warmly and offer assistance.")

if __name__ == "__main__":
    cli.run_app(server)