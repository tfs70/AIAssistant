import os

from module.rag_module import (
    get_or_build_vector_store,
    load_text_file,
    retrieve_context,
    split_documents,
)


TEST_FILE_PATH: str = "data/test_rag.txt"
CHROMA_DIRECTORY: str = "data/chroma_db"


def test_rag() -> None:
    """Test RAG retrieval"""

    if os.path.exists(path=CHROMA_DIRECTORY):
        import shutil

        shutil.rmtree(
            path=CHROMA_DIRECTORY,
        )

    documents = load_text_file(
        file_path=TEST_FILE_PATH,
    )

    assert len(documents) > 0

    print()
    print("===== DOCUMENTS =====")
    print(f"Documents: {len(documents)}")

    chunks = split_documents(
        documents=documents,
    )

    assert len(chunks) > 0

    print()
    print("===== CHUNKS =====")
    print(f"Chunks: {len(chunks)}")

    vector_store = get_or_build_vector_store(
        chunks=chunks,
    )

    query: str = "میزان فروش شرکت در ماه جاری چقدر است؟"

    results = retrieve_context(
        query=query,
        vector_store=vector_store,
        k=2,
    )

    assert len(results) > 0

    print()
    print("===== RETRIEVAL =====")
    print(f"Query: {query}")
    print()

    for index, result in enumerate(results):
        print(f"Result {index + 1}:")
        print(result.page_content)
        print()

    print("=====================")

    print()
    print("test_rag: PASSED")


if __name__ == "__main__":
    test_rag()