
import json
import os
from typing import Final
from datetime import datetime

from module.user_fact_model import UserFact


MEMORY_FILE_PATH: Final[str] = "data/long_memory.json"


def _read_memory() -> list[dict[str, str]]:
    """Read memory JSON, accepting UTF-8 files with or without BOM."""
    if not os.path.exists(MEMORY_FILE_PATH):
        return []

    with open(
        file=MEMORY_FILE_PATH,
        mode="r",
        encoding="utf-8-sig",
    ) as file:
        memory = json.load(file)

    if not isinstance(memory, list):
        raise ValueError(
            f"Memory file must contain a JSON list: {MEMORY_FILE_PATH}"
        )

    return memory


def _write_memory(memory: list[dict[str, str]]) -> None:
    """Write JSON as UTF-8 without BOM."""
    directory = os.path.dirname(MEMORY_FILE_PATH)
    os.makedirs(directory, exist_ok=True)

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


def save_fact(fact: UserFact) -> None:
    """Save a new fact or update an existing user's fact."""
    memory = _read_memory()

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
            if item.get("user_id") == fact.user_id
            and item.get("key") == fact.key
        ),
        None,
    )

    if existing_fact is not None:
        # Preserve the original creation timestamp.
        fact_data["created_at"] = existing_fact.get(
            "created_at",
            fact_data["created_at"],
        )
        existing_fact.update(fact_data)
    else:
        memory.append(fact_data)

    _write_memory(memory)


def get_user_facts(user_id: str) -> list[UserFact]:
    """Return all long-term facts belonging to one user."""
    memory = _read_memory()

    facts = [
        UserFact(
            user_id=item["user_id"],
            key=item["key"],
            value=item["value"],
            created_at=datetime.fromisoformat(item["created_at"]),
            updated_at=datetime.fromisoformat(item["updated_at"]),
        )
        for item in memory
        if item.get("user_id") == user_id
    ]

    facts.sort(key=lambda fact: fact.updated_at)
    return facts


def update_fact(fact: UserFact) -> None:
    """Update an existing fact or save it if it does not exist."""
    save_fact(fact)


def delete_fact(user_id: str, key: str) -> None:
    """Delete one fact belonging to the specified user."""
    memory = _read_memory()

    updated_memory = [
        item
        for item in memory
        if not (
            item.get("user_id") == user_id
            and item.get("key") == key
        )
    ]

    if len(updated_memory) != len(memory):
        _write_memory(updated_memory)
