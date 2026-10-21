from datetime import datetime, timedelta
import json

import pytest

import module.long_memory_module as memory_module
from module.user_fact_model import UserFact


@pytest.fixture
def memory_file(tmp_path, monkeypatch):
    """Use an isolated temporary JSON file for every test."""
    path = tmp_path / "long_memory.json"
    monkeypatch.setattr(memory_module, "MEMORY_FILE_PATH", str(path))
    return path


def make_fact(
    user_id: str = "USER_A",
    key: str = "name",
    value: str = "علی",
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> UserFact:
    now = datetime.now()
    return UserFact(
        user_id=user_id,
        key=key,
        value=value,
        created_at=created_at or now,
        updated_at=updated_at or now,
    )


def test_get_user_facts_returns_empty_when_file_does_not_exist(memory_file):
    assert memory_module.get_user_facts("USER_A") == []


def test_save_and_get_fact_round_trip(memory_file):
    fact = make_fact()

    memory_module.save_fact(fact)
    result = memory_module.get_user_facts("USER_A")

    assert len(result) == 1
    assert result[0].user_id == "USER_A"
    assert result[0].key == "name"
    assert result[0].value == "علی"


def test_facts_are_isolated_between_users(memory_file):
    memory_module.save_fact(make_fact(user_id="USER_A", value="علی"))
    memory_module.save_fact(make_fact(user_id="USER_B", value="رضا"))

    facts_a = memory_module.get_user_facts("USER_A")
    facts_b = memory_module.get_user_facts("USER_B")

    assert [(fact.key, fact.value) for fact in facts_a] == [("name", "علی")]
    assert [(fact.key, fact.value) for fact in facts_b] == [("name", "رضا")]


def test_saving_same_user_and_key_updates_instead_of_duplicating(memory_file):
    original_created_at = datetime(2026, 1, 1, 10, 0, 0)
    first = make_fact(
        value="علی",
        created_at=original_created_at,
        updated_at=original_created_at,
    )
    second_updated_at = original_created_at + timedelta(days=1)
    second = make_fact(
        value="علی‌رضا",
        created_at=second_updated_at,
        updated_at=second_updated_at,
    )

    memory_module.save_fact(first)
    memory_module.save_fact(second)

    facts = memory_module.get_user_facts("USER_A")

    assert len(facts) == 1
    assert facts[0].value == "علی‌رضا"
    # Updating a fact should not rewrite when it was first created.
    assert facts[0].created_at == original_created_at
    assert facts[0].updated_at == second_updated_at


def test_delete_fact_only_deletes_matching_users_fact(memory_file):
    memory_module.save_fact(make_fact(user_id="USER_A", key="name", value="علی"))
    memory_module.save_fact(make_fact(user_id="USER_A", key="job", value="برنامه‌نویس"))
    memory_module.save_fact(make_fact(user_id="USER_B", key="name", value="رضا"))

    memory_module.delete_fact(user_id="USER_A", key="name")

    assert [(f.key, f.value) for f in memory_module.get_user_facts("USER_A")] == [
        ("job", "برنامه‌نویس")
    ]
    assert [(f.key, f.value) for f in memory_module.get_user_facts("USER_B")] == [
        ("name", "رضا")
    ]


def test_read_accepts_utf8_bom_file(memory_file):
    memory = [
        {
            "user_id": "USER_A",
            "key": "name",
            "value": "علی",
            "created_at": "2026-10-21T10:00:00",
            "updated_at": "2026-10-21T10:00:00",
        }
    ]
    memory_file.write_text(
        json.dumps(memory, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )

    facts = memory_module.get_user_facts("USER_A")

    assert len(facts) == 1
    assert facts[0].value == "علی"


def test_write_uses_utf8_without_bom(memory_file):
    memory_module.save_fact(make_fact())

    raw = memory_file.read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf")
    assert json.loads(raw.decode("utf-8"))[0]["value"] == "علی"


def test_delete_fact_is_noop_when_file_does_not_exist(memory_file):
    memory_module.delete_fact(user_id="USER_A", key="name")
    assert not memory_file.exists()
