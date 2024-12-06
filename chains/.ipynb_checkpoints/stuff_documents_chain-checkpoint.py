import logging
from typing import List
from my_rag.chains.base_chain import BaseChain
from my_rag.prompts.prompt_template import PromptTemplate
from my_rag.llms.base_llm import BaseLLM
from my_rag.vectorstores.in_memory_vectorstore import Document

logger = logging.getLogger(__name__)

class StuffDocumentsChain(BaseChain):
    """
    A chain that simply "stuffs" documents into a prompt and queries the LLM.
    """

    def __init__(self, llm: BaseLLM, prompt: PromptTemplate):
        self.llm = llm
        self.prompt = prompt

    def run(self, query: str, docs: List[Document]) -> str:
        try:
            logger.info("Running StuffDocumentsChain with %d documents", len(docs))
            context = "\n".join([d.page_content for d in docs])
            formatted_prompt = self.prompt.format(context=context, query=query)
            response = self.llm.generate_response(formatted_prompt)
            return response
        except Exception as e:
            logger.error("Error running chain: %s", str(e))
            return "An error occurred while generating a response."
