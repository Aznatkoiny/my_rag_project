import abc
from typing import List

class BaseEmbeddings(abc.ABC):
    """
    Abstract base class for embedding models.
    """

    @abc.abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts and return their vector representations.
        """
        pass
