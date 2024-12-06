import logging
from typing import List

from my_rag.embeddings.openai_embeddings import OpenAIEmbeddings
from my_rag.vectorstores.in_memory_vectorstore import InMemoryVectorStore
from my_rag.transformers.long_context_reorder import LongContextReorder
from my_rag.prompts.prompt_template import PromptTemplate
from my_rag.llms.chat_openai import ChatOpenAI
from my_rag.chains.stuff_documents_chain import StuffDocumentsChain
from my_rag.config import DEFAULT_NUM_RETRIEVALS

logger = logging.getLogger(__name__)

def main():
    """
    Example usage demonstrating:
    - Embedding documents
    - Retrieving documents from a vector store
    - Applying a transformer to mitigate the "lost in the middle" effect
    - Passing transformed documents to a chain for Q&A
    """
    # Example documents
    texts = [
        "Basquetball is a great sport.",
        "Fly me to the moon is one of my favourite songs.",
        "The Celtics are my favourite team.",
        "This is a document about the Boston Celtics",
        "I simply love going to the movies",
        "The Boston Celtics won the game by 20 points",
        "This is just a random text.",
        "Elden Ring is one of the best games in the last 15 years.",
        "L. Kornet is one of the best Celtics players.",
        "Larry Bird was an iconic NBA player."
    ]

    query = "What can you tell me about the Celtics?"

    try:
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vector_store = InMemoryVectorStore.from_texts(texts, embedding=embeddings)

        # Retrieve documents
        docs = vector_store.retrieve(query, k=DEFAULT_NUM_RETRIEVALS)

        # Apply transformer to reorder documents
        transformer = LongContextReorder()
        reordered_docs = transformer.transform_documents(docs)

        # Create an LLM and prompt
        llm = ChatOpenAI(model_name="gpt-4")
        prompt_template = """
        Given these texts:
        -----
        {context}
        -----
        Please answer the following question:
        {query}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

        # Create a chain and run
        chain = StuffDocumentsChain(llm=llm, prompt=prompt)
        answer = chain.run(query, reordered_docs)

        logger.info("Final Answer:\n%s", answer)

    except Exception as e:
        logger.error("An error occurred during the retrieval or QA process: %s", str(e))

if __name__ == "__main__":
    main()
