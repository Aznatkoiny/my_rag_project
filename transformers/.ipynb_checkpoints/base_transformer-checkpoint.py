import abc
from typing import List
from my_rag.vectorstores.in_memory_vectorstore import Document

class BaseDocumentTransformer(abc.ABC):
    """
    Abstract base class for document transformers.
    Transformers modify or reorder a list of retrieved documents before passing them to the LLM.
    """

    @abc.abstractmethod
    def transform_documents(self, docs: List[Document]) -> List[Document]:
        """
        Transform a list of documents, e.g., by reordering, filtering, etc.
        """
        pass
