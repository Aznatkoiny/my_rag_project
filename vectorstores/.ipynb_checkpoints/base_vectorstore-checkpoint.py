import abc
from typing import List, Any

class BaseVectorStore(abc.ABC):
    """
    Abstract base class for vector stores.
    """

    @abc.abstractmethod
    def add_texts(self, texts: List[str]):
        """
        Add texts to the vector store.
        """
        pass

    @abc.abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[Any]:
        """
        Retrieve the top-k most relevant documents for the given query.
        """
        pass
