from typing import Final

from langchain_ollama import OllamaEmbeddings


EMBEDDING_MODEL_NAME: Final[str] = "qwen3-embedding:0.6b"


def test_embedding_multiple() -> None:
    """Test multiple embeddings"""

    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )

    texts: list[str] = [
        "فروش این ماه شرکت چقدر بوده است؟",
        "میزان فروش شرکت در ماه جاری چقدر است؟",
        "موجودی انبار شرکت چقدر است؟",
    ]

    embeddings: list[list[float]] = embedding.embed_documents(
        texts=texts,
    )

    print()
    print("===== MULTIPLE EMBEDDINGS =====")

    for index, result in enumerate(embeddings):
        print(
            f"Text {index + 1}: "
            f"dimensions={len(result)}"
        )

    print("===============================")

    assert len(embeddings) == 3

    assert len(embeddings[0]) > 0
    assert len(embeddings[1]) > 0
    assert len(embeddings[2]) > 0

    print()
    print("test_embedding_multiple: PASSED")


if __name__ == "__main__":
    test_embedding_multiple()