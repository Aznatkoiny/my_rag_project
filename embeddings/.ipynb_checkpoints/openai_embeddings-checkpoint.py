import os
import logging
from typing import List

import openai
from my_rag.embeddings.base_embeddings import BaseEmbeddings

logger = logging.getLogger(__name__)

class OpenAIEmbeddings(BaseEmbeddings):
    """
    Concrete embeddings class using OpenAI Embeddings API.
    """

    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables.")
            raise ValueError("OPENAI_API_KEY not set.")
        openai.api_key = api_key

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts using OpenAI Embeddings.
        """
        try:
            logger.info("Requesting embeddings for %d texts", len(texts))
            response = openai.Embedding.create(
                input=texts,
                engine=self.model_name
            )
            embeddings = [data["embedding"] for data in response["data"]]
            return embeddings
        except Exception as e:
            logger.error("Failed to get embeddings: %s", str(e))
            raise
