# My RAG (Retriever-Augmented Generation) Example

This codebase demonstrates a production-quality, object-oriented approach to mitigating the "lost in the middle" effect when providing long context inputs to GPT-4.

## Overview

- **Object-Oriented Design:**  
  The codebase uses abstract base classes and concrete implementations for embeddings, vector stores, transformers, LLMs, and chains.  
  The "lost in the middle" mitigation is implemented as a document transformer that can be easily replaced or extended.

- **Modular Codebase:**  
  Classes and logic are separated into modules, grouped by functionality.

- **Documentation & Logging:**  
  Comprehensive docstrings are provided, and `logging` is used throughout.

- **Exception Handling:**  
  Potential errors (e.g., retrieval issues, API failures) are handled gracefully.

- **"Lost in the Middle" Mitigation:**  
  Implemented via a `LongContextReorder` transformer. Other strategies can be added by extending `BaseDocumentTransformer`.

## Usage

1. Install dependencies:
   - `pip install openai`
   - Set your OpenAI API key as an environment variable: `export OPENAI_API_KEY=your_key_here`.

2. Run the example:
   ```bash
   python main.py
