"""Defaults aligned with test.ipynb."""

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 2
RETRIEVER_SEARCH_TYPE = "similarity"
TRANSCRIPT_LANGUAGES = ("en",)
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2

QA_PROMPT_TEMPLATE = """
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """
