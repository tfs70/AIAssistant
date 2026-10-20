
import hashlib
import shutil
from pathlib import Path
from typing import Final

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from module.embedding_module import EMBEDDING_MODEL_NAME


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DOCUMENTS_DIRECTORY: Final[Path] = PROJECT_ROOT / "documents"
PERSIST_DIRECTORY: Final[Path] = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME: Final[str] = "rag_documents"

CHUNK_SIZE: Final[int] = 900
CHUNK_OVERLAP: Final[int] = 150


def create_vector_store(
    collection_name: str = COLLECTION_NAME,
    persist_directory: str | Path = PERSIST_DIRECTORY,
) -> Chroma:
    """Create or open the persistent Chroma collection."""

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

    return Chroma(
        embedding_function=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )


def load_pdf(file_path: Path) -> list[Document]:
    """Extract text from a PDF while retaining page metadata."""

    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    relative_path = file_path.relative_to(DOCUMENTS_DIRECTORY).as_posix()

    for page in pages:
        page.metadata["source"] = str(file_path)
        page.metadata["relative_source"] = relative_path
        page.metadata["file_name"] = file_path.name
        page.metadata["page_number"] = int(page.metadata.get("page", 0)) + 1

    return pages


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """Split PDF pages into searchable chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", "؛", "،", " ", ""],
    )

    return splitter.split_documents(documents)


def rebuild_vector_store() -> dict[str, int]:
    """Delete the existing RAG index and rebuild it from all PDFs."""

    if not DOCUMENTS_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIRECTORY}"
        )

    pdf_files = sorted(
        path
        for path in DOCUMENTS_DIRECTORY.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found under: {DOCUMENTS_DIRECTORY}"
        )

    print("\n===== RAG REBUILD =====")
    print(f"Documents directory: {DOCUMENTS_DIRECTORY}")
    print(f"PDF files found: {len(pdf_files)}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")

    # Extract and split all PDFs before deleting the existing index.
    all_chunks: list[Document] = []
    total_pages = 0
    failed_files: list[str] = []

    for pdf_file in pdf_files:
        try:
            pages = load_pdf(pdf_file)
            total_pages += len(pages)

            nonempty_pages = [
                page for page in pages if page.page_content.strip()
            ]

            if not nonempty_pages:
                failed_files.append(
                    f"{pdf_file}: no extractable text"
                )
                continue

            chunks = split_documents(nonempty_pages)

            for index, chunk in enumerate(chunks):
                identity = (
                    f"{chunk.metadata['relative_source']}:"
                    f"{chunk.metadata['page_number']}:"
                    f"{index}:"
                    f"{chunk.page_content}"
                )
                chunk.metadata["chunk_id"] = hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()

            all_chunks.extend(chunks)

            print(
                f"[OK] {pdf_file.relative_to(DOCUMENTS_DIRECTORY)} "
                f"| pages={len(pages)} "
                f"| chunks={len(chunks)}"
            )

        except Exception as exc:
            failed_files.append(f"{pdf_file}: {exc}")
            print(f"[ERROR] {pdf_file}: {exc}")

    if not all_chunks:
        raise RuntimeError(
            "No text chunks could be extracted. "
            "The existing Chroma index has not been deleted."
        )

    if failed_files:
        print("\nFiles with extraction errors:")
        for error in failed_files:
            print(f"- {error}")

        raise RuntimeError(
            f"{len(failed_files)} PDF file(s) failed processing. "
            "The existing Chroma index has not been deleted. "
            "Fix the errors before rebuilding."
        )

    # Destructive operation: remove the old persistent index only
    # after PDF extraction and chunking have succeeded.
    if PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)

    # Recreate the collection and persist all chunks.
    vector_store = create_vector_store()

    batch_size = 100

    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start:start + batch_size]
        ids = [
            chunk.metadata["chunk_id"]
            for chunk in batch
        ]
        vector_store.add_documents(
            documents=batch,
            ids=ids,
        )

        print(
            f"Indexed chunks: "
            f"{min(start + len(batch), len(all_chunks))}"
            f"/{len(all_chunks)}"
        )

    indexed_count = vector_store._collection.count()

    if indexed_count != len(all_chunks):
        raise RuntimeError(
            f"Index count mismatch: expected {len(all_chunks)}, "
            f"found {indexed_count}."
        )

    summary = {
        "pdf_files": len(pdf_files),
        "pages": total_pages,
        "chunks": len(all_chunks),
        "indexed": indexed_count,
    }

    print("\n===== REBUILD COMPLETED =====")
    print(f"PDF files: {summary['pdf_files']}")
    print(f"Pages: {summary['pages']}")
    print(f"Chunks created: {summary['chunks']}")
    print(f"Chunks indexed: {summary['indexed']}")

    return summary


def retrieve_context(
    query: str,
    k: int = 5,
    collection_name: str = COLLECTION_NAME,
    persist_directory: str | Path = PERSIST_DIRECTORY,
) -> str:
    """Retrieve relevant chunks with their source and page information."""

    if not query.strip():
        return ""

    vector_store = create_vector_store(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    results = vector_store.similarity_search(
        query=query,
        k=k,
    )

    sections: list[str] = []

    for index, document in enumerate(results, start=1):
        source = document.metadata.get(
            "relative_source",
            document.metadata.get("source", "Unknown"),
        )
        page = document.metadata.get("page_number", "Unknown")

        sections.append(
            f"[Source {index}: {source}, page {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(sections)
