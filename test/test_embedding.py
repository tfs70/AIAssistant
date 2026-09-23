from module.embedding_module import create_embedding


def test_embedding() -> None:
    """Test embedding"""

    text: str = "فروش این ماه شرکت چقدر بوده است؟"

    embedding: list[float] = create_embedding(
        text=text,
    )

    print()
    print("===== EMBEDDING =====")
    print(f"Text: {text}")
    print(f"Dimensions: {len(embedding)}")
    print(f"First values: {embedding[:10]}")
    print("=====================")

    assert len(embedding) > 0

    print()
    print("test_embedding: PASSED")


if __name__ == "__main__":
    test_embedding()