# app/services/openai_service.py
import openai
import time
import logging
import tiktoken
from app.core.config import settings

class OpenAIService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenAIService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        openai.api_key = settings.OPENAI_API_KEY
        self._initialized = True

    def create_embeddings(self, text: str, model: str = "text-embedding-ada-002"):
        try:
            response = openai.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise
