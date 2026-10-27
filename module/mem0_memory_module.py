from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from module.embedding_module import EMBEDDING_MODEL_NAME


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MEM0_CHROMA_PATH = Path(
    os.getenv(
        "PERSIS_MEM0_CHROMA_PATH",
        str(PROJECT_ROOT / "data" / "mem0_chroma_db"),
    )
)

MEM0_COLLECTION_NAME = os.getenv(
    "PERSIS_MEM0_COLLECTION_NAME",
    "mem0_user_memory",
)

MEM0_LLM_MODEL = os.getenv(
    "PERSIS_MEM0_LLM_MODEL",
    "qwen3.5:4b",
)

OLLAMA_BASE_URL = os.getenv(
    "PERSIS_OLLAMA_BASE_URL",
    "http://localhost:11434",
)

EMBEDDING_DIMS = 1024
METADATA_SOURCE = "persis_mem0_pilot"


def build_mem0_config() -> dict[str, Any]:
    """Build a fully local Mem0 configuration."""
    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": MEM0_COLLECTION_NAME,
                "path": str(MEM0_CHROMA_PATH),
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": MEM0_LLM_MODEL,
                "temperature": 0,
                "ollama_base_url": OLLAMA_BASE_URL,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": EMBEDDING_MODEL_NAME,
                "embedding_dims": EMBEDDING_DIMS,
                "ollama_base_url": OLLAMA_BASE_URL,
            },
        },
        "history_db_path": str(
            PROJECT_ROOT / "data" / "mem0_history.db"
        ),
    }


def _new_memory() -> Any:
    """Create a Mem0 instance using the isolated local store."""
    try:
        from mem0 import Memory
    except ImportError as exc:
        raise RuntimeError(
            "Mem0 dependencies are missing. "
            "Install the project's reviewed Mem0 requirements."
        ) from exc

    MEM0_CHROMA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )
    (PROJECT_ROOT / "data").mkdir(
        parents=True,
        exist_ok=True,
    )

    return Memory.from_config(build_mem0_config())


def _validate_user_id(user_id: str) -> None:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")


def add_memory(
    user_id: str,
    text: str,
    *,
    key: str | None = None,
) -> dict[str, Any]:
    """Add a memory for a specific user."""
    _validate_user_id(user_id)

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")

    metadata: dict[str, str] = {
        "source": METADATA_SOURCE,
    }

    if key is not None and key.strip():
        metadata["key"] = key.strip().lower()

    print(
        f"[Mem0] Creating instance for {user_id}",
        flush=True,
    )
    memory = _new_memory()

    print(
        f"[Mem0] Instance created for {user_id}",
        flush=True,
    )
    print(
        f"[Mem0] Calling add for {user_id}",
        flush=True,
    )

    result = memory.add(
        text.strip(),
        user_id=user_id,
        metadata=metadata,
    )

    print(
        f"[Mem0] Add result for {user_id}: {result!r}",
        flush=True,
    )

    return result if isinstance(result, dict) else {
        "result": result
    }


def search_memories(
    user_id: str,
    query: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search memories restricted to one user."""
    _validate_user_id(user_id)

    if not isinstance(query, str) or not query.strip():
        return []

    if limit < 1:
        raise ValueError("limit must be at least 1")

    print(
        f"[Mem0] Searching memories for {user_id}",
        flush=True,
    )

    result = _new_memory().search(
        query=query.strip(),
        filters={"user_id": user_id},
        top_k=limit,
    )

    print(
        f"[Mem0] Search result for {user_id}: {result!r}",
        flush=True,
    )

    if isinstance(result, dict):
        items = result.get("results", [])
        return items if isinstance(items, list) else []

    return result if isinstance(result, list) else []


def get_all_memories(
    user_id: str,
) -> list[dict[str, Any]]:
    """Retrieve memories restricted to one user."""
    _validate_user_id(user_id)

    print(
        f"[Mem0] Getting all memories for {user_id}",
        flush=True,
    )

    result = _new_memory().get_all(
        filters={"user_id": user_id},
    )

    print(
        f"[Mem0] get_all result for {user_id}: {result!r}",
        flush=True,
    )

    if isinstance(result, dict):
        items = result.get("results", [])
        return items if isinstance(items, list) else []

    return result if isinstance(result, list) else []


def delete_memory(memory_id: str) -> Any:
    """Delete one memory by its ID."""
    if not isinstance(memory_id, str) or not memory_id.strip():
        raise ValueError("memory_id must not be empty")

    return _new_memory().delete(
        memory_id=memory_id,
    )


def delete_user_memories(
    user_id: str,
) -> Any:
    """Delete all Mem0 memories belonging to one user."""
    _validate_user_id(user_id)

    print(
        f"[Mem0] Deleting memories for {user_id}",
        flush=True,
    )

    result = _new_memory().delete_all(
        user_id=user_id,
    )

    print(
        f"[Mem0] Delete result for {user_id}: {result!r}",
        flush=True,
    )

    return result


def replace_keyed_memory(
    user_id: str,
    key: str,
    text: str,
) -> dict[str, Any]:
    """Replace memories with the same logical key for one user."""
    _validate_user_id(user_id)

    if not isinstance(key, str) or not key.strip():
        raise ValueError("key must not be empty")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")

    normalized_key = key.strip().lower()
    memory = _new_memory()

    existing = memory.get_all(
        filters={"user_id": user_id},
    )

    if isinstance(existing, dict):
        items = existing.get("results", [])
    elif isinstance(existing, list):
        items = existing
    else:
        items = []

    deleted_ids: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata") or {}

        if metadata.get("key") != normalized_key:
            continue

        memory_id = item.get("id")

        if memory_id:
            memory.delete(memory_id=memory_id)
            deleted_ids.append(str(memory_id))

    print(
        f"[Mem0] Adding replacement memory for {user_id}",
        flush=True,
    )

    add_result = memory.add(
        text.strip(),
        user_id=user_id,
        metadata={
            "source": METADATA_SOURCE,
            "key": normalized_key,
        },
    )

    print(
        f"[Mem0] Replacement add result: {add_result!r}",
        flush=True,
    )

    return {
        "deleted_ids": deleted_ids,
        "add_result": add_result,
    }