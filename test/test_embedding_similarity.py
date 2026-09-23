from math import sqrt

from module.embedding_module import create_embedding


def calculate_similarity(
    first: list[float],
    second: list[float],
) -> float:
    """Calculate cosine similarity"""

    dot_product: float = sum(
        first_value * second_value
        for first_value, second_value in zip(first, second)
    )

    first_norm: float = sqrt(
        sum(
            value * value
            for value in first
        )
    )

    second_norm: float = sqrt(
        sum(
            value * value
            for value in second
        )
    )

    if first_norm == 0 or second_norm == 0:
        return 0.0

    return dot_product / (first_norm * second_norm)


def test_embedding_similarity() -> None:
    """Test embedding similarity"""

    text_1: str = "فروش این ماه شرکت چقدر بوده است؟"
    text_2: str = "میزان فروش شرکت در ماه جاری چقدر است؟"
    text_3: str = "موجودی انبار شرکت چقدر است؟"

    embedding_1: list[float] = create_embedding(
        text=text_1,
    )

    embedding_2: list[float] = create_embedding(
        text=text_2,
    )

    embedding_3: list[float] = create_embedding(
        text=text_3,
    )

    similarity_1_2: float = calculate_similarity(
        first=embedding_1,
        second=embedding_2,
    )

    similarity_1_3: float = calculate_similarity(
        first=embedding_1,
        second=embedding_3,
    )

    print()
    print("===== EMBEDDING SIMILARITY =====")
    print(f"Text 1: {text_1}")
    print(f"Text 2: {text_2}")
    print(f"Text 3: {text_3}")
    print()
    print(f"Similarity 1 -> 2: {similarity_1_2:.6f}")
    print(f"Similarity 1 -> 3: {similarity_1_3:.6f}")
    print("================================")

    assert similarity_1_2 > similarity_1_3

    print()
    print("test_embedding_similarity: PASSED")


if __name__ == "__main__":
    test_embedding_similarity()