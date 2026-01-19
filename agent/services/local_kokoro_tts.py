"""
Local Kokoro TTS Service wrapper for Pipecat.
Connects to local Kokoro TTS instance via OpenAI-compatible API.
"""

import os
from loguru import logger
from pipecat.services.openai import OpenAITTSService


class LocalKokoroTTSService(OpenAITTSService):
    """
    Kokoro TTS service that connects to a local Kokoro instance
    running on a custom URL with OpenAI-compatible API.
    
    Supports voice switching for bilingual translation:
    - English voices: af_nova, af_nicole, etc.
    - Spanish voices: ef_dora, ef_bella, etc.
    """

    def __init__(
        self,
        *,
        base_url: str = None,
        model: str = "tts-1",
        voice: str = "af_nova",
        api_key: str = "not-needed",
        **kwargs
    ):
        """
        Initialize Local Kokoro TTS service.
        
        Args:
            base_url: URL of local Kokoro service (e.g., http://localhost:8880/v1)
            model: Model identifier (default: tts-1)
            voice: Voice to use (default: af_nova for English)
            api_key: API key (not needed for local, but required by OpenAI client)
        """
        base_url = base_url or os.getenv("KOKORO_URL", "http://localhost:8880/v1")
        
        logger.info(f"Initializing LocalKokoroTTSService with base_url: {base_url}, voice: {voice}")
        
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            voice=voice,
            **kwargs
        )
    
    def set_voice(self, voice: str):
        """
        Dynamically change the TTS voice.
        
        Args:
            voice: Voice identifier (e.g., 'af_nova', 'ef_dora')
        """
        logger.info(f"Switching TTS voice to: {voice}")
        self._voice = voice
