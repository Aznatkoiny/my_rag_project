import os
import logging
import openai
from my_rag.llms.base_llm import BaseLLM

logger = logging.getLogger(__name__)

class ChatOpenAI(BaseLLM):
    """
    A wrapper around OpenAI's chat-based endpoints.
    """

    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables.")
            raise ValueError("OPENAI_API_KEY not set.")
        openai.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the Chat API.
        """
        try:
            logger.info("Sending prompt to LLM (model: %s)", self.model_name)
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages
            )
            answer = response["choices"][0]["message"]["content"].strip()
            logger.info("Received response from LLM.")
            return answer
        except Exception as e:
            logger.error("Error generating response from LLM: %s", str(e))
            return "An error occurred while querying the LLM."
