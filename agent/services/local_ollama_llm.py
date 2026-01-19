"""
Local Ollama LLM Service wrapper for Pipecat.
Connects to local Ollama instance.
"""

import os
from loguru import logger
from pipecat.services.openai import OpenAILLMService


class LocalOllamaLLMService(OpenAILLMService):
    """
    Ollama LLM service that connects to a local Ollama instance
    with OpenAI-compatible API.
    """

    def __init__(
        self,
        *,
        base_url: str = None,
        model: str = "gemma3:1b",
        api_key: str = "not-needed",
        **kwargs
    ):
        """
        Initialize Local Ollama LLM service.
        
        Args:
            base_url: URL of local Ollama service (e.g., http://localhost:11434/v1)
            model: Model identifier (default: gemma3:1b)
            api_key: API key (not needed for local, but required by OpenAI client)
        """
        base_url = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
        
        logger.info(f"Initializing LocalOllamaLLMService with base_url: {base_url}, model: {model}")
        
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs
        )
