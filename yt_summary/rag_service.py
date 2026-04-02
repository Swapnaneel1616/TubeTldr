"""Chunking, embeddings, and FAISS store (test.ipynb cells 2, 4, 6)."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def build_vector_store(full_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents([full_text])
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)


def retriever_from_store(vector_store):
    return vector_store.as_retriever(
        search_type=config.RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": config.RETRIEVER_K},
    )
