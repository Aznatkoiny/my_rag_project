import abc
from typing import List, Any
from my_rag.vectorstores.in_memory_vectorstore import Document

class BaseChain(abc.ABC):
    """
    Abstract base class for chains.
    A chain orchestrates the combination of documents and prompts, calling an LLM, etc.
    """

    @abc.abstractmethod
    def run(self, query: str, docs: List[Document]) -> Any:
        """
        Execute the chain logic and return a response.
        """
        pass
