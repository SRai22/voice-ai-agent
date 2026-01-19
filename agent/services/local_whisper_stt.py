"""
Local Whisper STT Service wrapper for Pipecat.
Connects to local Whisper instance via OpenAI-compatible API.
"""

import os
from loguru import logger
from pipecat.services.openai import OpenAISTTService


class LocalWhisperSTTService(OpenAISTTService):
    """
    Whisper STT service that connects to a local Whisper instance
    running on a custom URL with OpenAI-compatible API.
    """

    def __init__(
        self,
        *,
        base_url: str = None,
        model: str = "Systran/faster-whisper-small",
        api_key: str = "not-needed",
        **kwargs
    ):
        """
        Initialize Local Whisper STT service.
        
        Args:
            base_url: URL of local Whisper service (e.g., http://localhost:11435/v1)
            model: Model identifier (default: Systran/faster-whisper-small)
            api_key: API key (not needed for local, but required by OpenAI client)
        """
        base_url = base_url or os.getenv("WHISPER_URL", "http://localhost:11435/v1")
        
        logger.info(f"Initializing LocalWhisperSTTService with base_url: {base_url}, model: {model}")
        
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )
