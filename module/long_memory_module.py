import json
import os
from datetime import datetime
from typing import Final

from module.user_fact_model import UserFact


MEMORY_FILE_PATH: Final[str] = "data/long_memory.json"


def save_fact(
    fact: UserFact,
) -> None:
    """ذخیره یا بروزرسانی یک UserFact"""

    os.makedirs(
        name=os.path.dirname(MEMORY_FILE_PATH),
        exist_ok=True,
    )

    memory: list[dict[str, str]] = []

    if os.path.exists(path=MEMORY_FILE_PATH):
        with open(
            file=MEMORY_FILE_PATH,
            mode="r",
            encoding="utf-8",
        ) as file:
            memory = json.load(file)

    fact_data: dict[str, str] = {
        "user_id": fact.user_id,
        "key": fact.key,
        "value": fact.value,
        "created_at": fact.created_at.isoformat(),
        "updated_at": fact.updated_at.isoformat(),
    }

    existing_fact = next(
        (
            item
            for item in memory
            if item["user_id"] == fact.user_id
            and item["key"] == fact.key
        ),
        None,
    )

    if existing_fact:
        existing_fact.update(fact_data)
    else:
        memory.append(fact_data)

    with open(
        file=MEMORY_FILE_PATH,
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            memory,
            file,
            ensure_ascii=False,
            indent=4,
        )


def get_user_facts(
    user_id: str,
) -> list[UserFact]:
    """دریافت تمام Factهای یک کاربر"""

    if not os.path.exists(path=MEMORY_FILE_PATH):
        return []

    with open(
        file=MEMORY_FILE_PATH,
        mode="r",
        encoding="utf-8",
    ) as file:
        memory: list[dict[str, str]] = json.load(file)

    facts: list[UserFact] = [
        UserFact(
            user_id=fact["user_id"],
            key=fact["key"],
            value=fact["value"],
            created_at=datetime.fromisoformat(fact["created_at"]),
            updated_at=datetime.fromisoformat(fact["updated_at"]),
        )
        for fact in memory
        if fact["user_id"] == user_id
    ]

    facts.sort(
        key=lambda fact: fact.updated_at,
    )

    return facts


def update_fact(
    fact: UserFact,
) -> None:
    """بروزرسانی یک UserFact"""

    save_fact(
        fact=fact,
    )


def delete_fact(
    user_id: str,
    key: str,
) -> None:
    """حذف یک UserFact"""

    if not os.path.exists(path=MEMORY_FILE_PATH):
        return

    with open(
        file=MEMORY_FILE_PATH,
        mode="r",
        encoding="utf-8",
    ) as file:
        memory: list[dict[str, str]] = json.load(file)

    memory = [
        fact
        for fact in memory
        if not (
            fact["user_id"] == user_id
            and fact["key"] == key
        )
    ]

    with open(
        file=MEMORY_FILE_PATH,
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            memory,
            file,
            ensure_ascii=False,
            indent=4,
        )