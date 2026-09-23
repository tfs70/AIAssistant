from typing import Final

from langchain_ollama import OllamaEmbeddings


EMBEDDING_MODEL_NAME: Final[str] = "qwen3-embedding:0.6b"

embedding = OllamaEmbeddings(
    model=EMBEDDING_MODEL_NAME,
)


def create_embedding(
    text: str,
) -> list[float]:
    """Create embedding for text."""

    return embedding.embed_query(text=text)