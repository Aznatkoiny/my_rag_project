import abc

class BaseLLM(abc.ABC):
    """
    Abstract base class for language models.
    """

    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Given a prompt, generate a response from the language model.
        """
        pass
