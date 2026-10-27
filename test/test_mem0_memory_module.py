from unittest.mock import MagicMock

import pytest

import module.mem0_memory_module as mem0_module


def test_config_uses_separate_collection_and_local_providers(monkeypatch, tmp_path):
    monkeypatch.setattr(mem0_module, "MEM0_CHROMA_PATH", tmp_path / "mem0")
    config = mem0_module.build_mem0_config()

    assert config["vector_store"]["provider"] == "chroma"
    assert config["vector_store"]["config"]["collection_name"] == "mem0_user_memory"
    assert config["vector_store"]["config"]["path"] == str(tmp_path / "mem0")
    assert config["embedder"]["provider"] == "ollama"
    assert config["embedder"]["config"]["model"] == "qwen3-embedding:0.6b"
    assert config["embedder"]["config"]["embedding_dims"] == 1024
    assert config["llm"]["provider"] == "ollama"
    assert config["llm"]["config"]["model"] == "qwen3.5:4b"


def test_add_memory_is_scoped_and_tags_source(monkeypatch):
    fake = MagicMock()
    fake.add.return_value = {"results": [{"id": "m1"}]}
    monkeypatch.setattr(mem0_module, "_new_memory", lambda: fake)

    result = mem0_module.add_memory("USER_A", "اسم من علی است", key="name")

    assert result["results"][0]["id"] == "m1"
    fake.add.assert_called_once_with(
        "اسم من علی است",
        user_id="USER_A",
        metadata={"source": "persis_mem0_pilot", "key": "name"},
    )


def test_search_is_scoped_to_requested_user(monkeypatch):
    fake = MagicMock()
    fake.search.return_value = {"results": [{"id": "m1", "memory": "نام علی است"}]}
    monkeypatch.setattr(mem0_module, "_new_memory", lambda: fake)

    result = mem0_module.search_memories("USER_A", "اسم من چیست؟", limit=3)

    assert result == [{"id": "m1", "memory": "نام علی است"}]
    fake.search.assert_called_once_with(
        query="اسم من چیست؟",
        filters={"user_id": "USER_A"},
        top_k=3,
    )


@pytest.mark.parametrize("user_id", ["", "  ", None])
def test_empty_user_id_is_rejected(user_id):
    with pytest.raises(ValueError, match="user_id"):
        mem0_module.search_memories(user_id, "test")


def test_empty_query_returns_without_initializing_mem0(monkeypatch):
    def fail_if_called():
        raise AssertionError("Mem0 should not initialize for an empty query")

    monkeypatch.setattr(mem0_module, "_new_memory", fail_if_called)
    assert mem0_module.search_memories("USER_A", "  ") == []


def test_replace_keyed_memory_only_deletes_matching_key(monkeypatch):
    fake = MagicMock()
    fake.get_all.return_value = {
        "results": [
            {"id": "name-1", "metadata": {"key": "name"}},
            {"id": "job-1", "metadata": {"key": "job"}},
        ]
    }
    fake.add.return_value = {"results": [{"id": "name-2"}]}
    monkeypatch.setattr(mem0_module, "_new_memory", lambda: fake)

    result = mem0_module.replace_keyed_memory(
        "USER_A",
        "name",
        "اسم کاربر علی‌رضا است",
    )

    fake.delete.assert_called_once_with(memory_id="name-1")
    fake.add.assert_called_once_with(
        "اسم کاربر علی‌رضا است",
        user_id="USER_A",
        metadata={"source": "persis_mem0_pilot", "key": "name"},
    )
    assert result["deleted_ids"] == ["name-1"]


def test_delete_user_memories_requires_user_id(monkeypatch):
    with pytest.raises(ValueError, match="user_id"):
        mem0_module.delete_user_memories("")
