import logging
from typing import List
from my_rag.vectorstores.in_memory_vectorstore import Document
from my_rag.transformers.base_transformer import BaseDocumentTransformer

logger = logging.getLogger(__name__)

class LongContextReorder(BaseDocumentTransformer):
    """
    Reorders documents to mitigate the "lost in the middle" effect by placing
    highly relevant documents at the beginning and the end, and less relevant ones in the middle.
    Assumes documents are initially sorted by relevance in descending order.
    """

    def transform_documents(self, docs: List[Document]) -> List[Document]:
        try:
            # Docs come in descending relevance order:
            # We'll take the top half and place it at the start, and the bottom half at the end reversed.
            if len(docs) < 3:
                # Not enough documents to reorder meaningfully
                logger.info("Not enough documents to reorder, returning as is.")
                return docs

            half = len(docs) // 2
            first_half = docs[:half]
            second_half = docs[half:]

            # Reorder: first half normal at start, second half reversed at the end
            reordered = first_half + second_half[::-1]
            logger.info("Reordered documents to mitigate lost in the middle.")
            return reordered
        except Exception as e:
            logger.error("Error in long context reordering: %s", str(e))
            # On error, return original documents
            return docs
