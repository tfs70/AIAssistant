from typing import Final

from langchain_ollama import OllamaEmbeddings


EMBEDDING_MODEL_NAME: Final[str] = "qwen3-embedding:0.6b"


def create_embedding(
    text: str,
) -> list[float]:
    """Create embedding for text"""

    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )

    result: list[float] = embedding.embed_query(
        text=text,
    )

    return result