"""RunnableParallel RAG chain (test.ipynb cells 8–9, 15–20)."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from . import config


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def build_main_chain(retriever):
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    prompt = PromptTemplate(
        template=config.QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    parser = StrOutputParser()
    return parallel_chain | prompt | llm | parser
