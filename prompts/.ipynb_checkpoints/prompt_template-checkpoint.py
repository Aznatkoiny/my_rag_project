import logging
from typing import List

logger = logging.getLogger(__name__)

class PromptTemplate:
    """
    A simple prompt template that can be formatted with context and query.
    """

    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        try:
            logger.info("Formatting prompt with variables: %s", list(kwargs.keys()))
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error("Missing input variable for prompt: %s", str(e))
            raise
