import os
from datetime import datetime

from module.long_memory_module import (
    delete_fact,
    get_user_facts,
    save_fact,
    update_fact,
)
from module.user_fact_model import UserFact


MEMORY_FILE_PATH = "data/long_memory.json"


def test_long_memory() -> None:
    """تست Long Memory"""

    if os.path.exists(MEMORY_FILE_PATH):
        os.remove(MEMORY_FILE_PATH)

    now: datetime = datetime.now()

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    fact = UserFact(
        user_id="user_001",
        key="name",
        value="علی",
        created_at=now,
        updated_at=now,
    )

    save_fact(
        fact=fact,
    )

    facts = get_user_facts(
        user_id="user_001",
    )

    assert len(facts) == 1
    assert facts[0].key == "name"
    assert facts[0].value == "علی"

    # -------------------------------------------------
    # Update
    # -------------------------------------------------

    updated_fact = UserFact(
        user_id="user_001",
        key="name",
        value="علی برنامه‌نویس",
        created_at=now,
        updated_at=datetime.now(),
    )

    update_fact(
        fact=updated_fact,
    )

    facts = get_user_facts(
        user_id="user_001",
    )

    assert len(facts) == 1
    assert facts[0].value == "علی برنامه‌نویس"

    # -------------------------------------------------
    # Another user
    # -------------------------------------------------

    another_fact = UserFact(
        user_id="user_002",
        key="name",
        value="رضا",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    save_fact(
        fact=another_fact,
    )

    user_001_facts = get_user_facts(
        user_id="user_001",
    )

    user_002_facts = get_user_facts(
        user_id="user_002",
    )

    assert len(user_001_facts) == 1
    assert user_001_facts[0].value == "علی برنامه‌نویس"

    assert len(user_002_facts) == 1
    assert user_002_facts[0].value == "رضا"

    # -------------------------------------------------
    # Delete
    # -------------------------------------------------

    delete_fact(
        user_id="user_001",
        key="name",
    )

    user_001_facts = get_user_facts(
        user_id="user_001",
    )

    assert len(user_001_facts) == 0

    print()
    print("test_long_memory: PASSED")


if __name__ == "__main__":
    test_long_memory()