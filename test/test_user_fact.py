from datetime import datetime

from module.user_fact_model import UserFact


def test_user_fact() -> None:
    """تست UserFact"""

    now: datetime = datetime.now()

    fact = UserFact(
        user_id="user_001",
        key="name",
        value="علی",
        created_at=now,
        updated_at=now,
    )

    assert fact.user_id == "user_001"
    assert fact.key == "name"
    assert fact.value == "علی"
    assert fact.created_at == now
    assert fact.updated_at == now

    print()
    print("UserFact:")
    print(f"user_id: {fact.user_id}")
    print(f"key: {fact.key}")
    print(f"value: {fact.value}")
    print(f"created_at: {fact.created_at}")
    print(f"updated_at: {fact.updated_at}")

    print()
    print("test_user_fact: PASSED")


if __name__ == "__main__":
    test_user_fact()