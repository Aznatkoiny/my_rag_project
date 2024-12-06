import logging
from typing import List, NamedTuple

from my_rag.vectorstores.base_vectorstore import BaseVectorStore
from my_rag.embeddings.base_embeddings import BaseEmbeddings

logger = logging.getLogger(__name__)

class Document(NamedTuple):
    page_content: str
    embedding: List[float]

class InMemoryVectorStore(BaseVectorStore):
    """
    A simple in-memory vector store that retrieves documents by cosine similarity.
    """

    def __init__(self, embedding_model: BaseEmbeddings):
        self.embedding_model = embedding_model
        self.documents: List[Document] = []

    @classmethod
    def from_texts(cls, texts: List[str], embedding: BaseEmbeddings):
        instance = cls(embedding_model=embedding)
        instance.add_texts(texts)
        return instance

    def add_texts(self, texts: List[str]):
        embeddings = self.embedding_model.embed(texts)
        for text, emb in zip(texts, embeddings):
            self.documents.append(Document(page_content=text, embedding=emb))
        logger.info("Added %d documents to the vector store.", len(texts))

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        query_embedding = self.embedding_model.embed([query])[0]

        # Compute cosine similarity
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot = sum(x*y for x, y in zip(a, b))
            norm_a = sum(x*x for x in a) ** 0.5
            norm_b = sum(x*x for x in b) ** 0.5
            return dot / (norm_a * norm_b)

        # Rank by similarity
        scored = []
        for doc in self.documents:
            score = cosine_similarity(query_embedding, doc.embedding)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)

        retrieved = [d for _, d in scored[:k]]
        logger.info("Retrieved %d documents for query: '%s'", len(retrieved), query)
        return retrieved
