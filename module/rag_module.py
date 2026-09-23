import os

from typing import Final

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from module.embedding_module import EMBEDDING_MODEL_NAME


PERSIST_DIRECTORY: Final[str] = "data/chroma_db"
COLLECTION_NAME: Final[str] = "rag_documents"


def load_text_file(
    file_path: str,
) -> list[Document]:
    """Load text file"""

    loader = TextLoader(
        encoding="utf-8",
        file_path=file_path,
    )

    documents: list[Document] = loader.load()

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split documents"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Document] = splitter.split_documents(
        documents=documents,
    )

    return chunks


def get_or_build_vector_store(
    chunks: list[Document],
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = PERSIST_DIRECTORY,
) -> Chroma:
    """Get or build vector store"""

    from langchain_ollama import OllamaEmbeddings

    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )

    vector_store = Chroma(
        embedding_function=embedding,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    if len(chunks) > 0:
        vector_store.add_documents(
            documents=chunks,
        )

    return vector_store


def retrieve_context(
    query: str,
    vector_store: Chroma,
    k: int = 3,
) -> list[Document]:
    """Retrieve relevant documents"""

    results: list[Document] = vector_store.similarity_search(
        query=query,
        k=k,
    )

    return results