from typing import Final

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

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


def create_vector_store(
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = PERSIST_DIRECTORY,
) -> Chroma:
    """Create or open Chroma vector store"""

    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )

    vector_store = Chroma(
        embedding_function=embedding,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    return vector_store


def add_documents(
    documents: list[Document],
    vector_store: Chroma,
) -> None:
    """Add documents to vector store"""

    if not documents:
        return

    vector_store.add_documents(
        documents=documents,
    )


def build_vector_store(
    file_path: str,
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = PERSIST_DIRECTORY,
) -> Chroma:
    """Load, split and store documents"""

    documents: list[Document] = load_text_file(
        file_path=file_path,
    )

    chunks: list[Document] = split_documents(
        documents=documents,
    )

    vector_store = create_vector_store(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    add_documents(
        documents=chunks,
        vector_store=vector_store,
    )

    return vector_store


def retrieve_context(
    query: str,
    k: int = 3,
    collection_name: str = COLLECTION_NAME,
    persist_directory: str = PERSIST_DIRECTORY,
) -> str:
    """Retrieve relevant context"""

    vector_store = create_vector_store(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    results: list[Document] = vector_store.similarity_search(
        query=query,
        k=k,
    )

    context: str = "\n\n".join(
        result.page_content
        for result in results
    )

    return context