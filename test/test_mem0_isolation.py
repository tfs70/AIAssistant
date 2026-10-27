from __future__ import annotations

from typing import Any

from module.mem0_memory_module import (
    add_memory,
    delete_user_memories,
    get_all_memories,
    search_memories,
)


# These IDs are reserved exclusively for this test.
USER_A = "MEM0_ISOLATION_QA_A"
USER_B = "MEM0_ISOLATION_QA_B"

FACT_A = (
    "کاربر آزمایشی A نامش آذر است "
    "و کد اختصاصی او ALPHA-7319 است."
)

FACT_B = (
    "کاربر آزمایشی B نامش بهرام است "
    "و کد اختصاصی او BRAVO-8426 است."
)


def _memory_text(item: dict[str, Any]) -> str:
    """Extract text from a Mem0 memory record."""
    value = item.get("memory", "")
    return value if isinstance(value, str) else str(value)


def _assert_user_scope(
    memories: list[dict[str, Any]],
    expected_user_id: str,
) -> None:
    """Ensure returned records belong to the expected user."""
    for item in memories:
        metadata = item.get("metadata") or {}
        actual_user_id = (
            item.get("user_id")
            or metadata.get("user_id")
        )

        if actual_user_id is not None:
            assert actual_user_id == expected_user_id, (
                f"Memory leaked across users: "
                f"expected={expected_user_id!r}, "
                f"actual={actual_user_id!r}, "
                f"record={item!r}"
            )


def test_mem0_memory_isolation_between_users() -> None:
    print(
        "[START] Mem0 user isolation test",
        flush=True,
    )

    try:
        # Only delete the two dedicated synthetic test users.
        print("[1] Cleaning USER_A", flush=True)
        delete_user_memories(USER_A)

        print("[2] Cleaning USER_B", flush=True)
        delete_user_memories(USER_B)

        # Add an independent fact for each user.
        print("[3] Adding memory for USER_A", flush=True)
        result_a = add_memory(
            USER_A,
            FACT_A,
            key="identity_test",
        )
        print(
            f"[4] USER_A add result: {result_a!r}",
            flush=True,
        )

        print("[5] Adding memory for USER_B", flush=True)
        result_b = add_memory(
            USER_B,
            FACT_B,
            key="identity_test",
        )
        print(
            f"[6] USER_B add result: {result_b!r}",
            flush=True,
        )

        # Verify stored records using user-scoped retrieval.
        print("[7] Reading USER_A memories", flush=True)
        memories_a = get_all_memories(USER_A)
        print(
            f"[8] USER_A memories: {memories_a!r}",
            flush=True,
        )

        print("[9] Reading USER_B memories", flush=True)
        memories_b = get_all_memories(USER_B)
        print(
            f"[10] USER_B memories: {memories_b!r}",
            flush=True,
        )

        assert memories_a, (
            "No memories found for USER_A. "
            f"add_result={result_a!r}"
        )
        assert memories_b, (
            "No memories found for USER_B. "
            f"add_result={result_b!r}"
        )

        _assert_user_scope(memories_a, USER_A)
        _assert_user_scope(memories_b, USER_B)

        text_a = "\n".join(
            _memory_text(item) for item in memories_a
        ).casefold()

        text_b = "\n".join(
            _memory_text(item) for item in memories_b
        ).casefold()

        # Verify each user's records do not contain the
        # other user's unique test code.
        assert "bravo-8426" not in text_a, (
            "USER_B's unique code appeared in USER_A memories."
        )
        assert "alpha-7319" not in text_b, (
            "USER_A's unique code appeared in USER_B memories."
        )

        # Verify semantic search is also scoped to each user.
        print("[11] Searching USER_A memories", flush=True)
        search_a = search_memories(
            USER_A,
            "کد اختصاصی کاربر آزمایشی چیست؟",
            limit=10,
        )
        print(
            f"[12] USER_A search result: {search_a!r}",
            flush=True,
        )

        print("[13] Searching USER_B memories", flush=True)
        search_b = search_memories(
            USER_B,
            "کد اختصاصی کاربر آزمایشی چیست؟",
            limit=10,
        )
        print(
            f"[14] USER_B search result: {search_b!r}",
            flush=True,
        )

        assert search_a, (
            "Semantic search returned no memories for USER_A."
        )
        assert search_b, (
            "Semantic search returned no memories for USER_B."
        )

        search_text_a = "\n".join(
            _memory_text(item) for item in search_a
        ).casefold()

        search_text_b = "\n".join(
            _memory_text(item) for item in search_b
        ).casefold()

        assert "bravo-8426" not in search_text_a, (
            "USER_B's unique code leaked into USER_A search."
        )
        assert "alpha-7319" not in search_text_b, (
            "USER_A's unique code leaked into USER_B search."
        )

        _assert_user_scope(search_a, USER_A)
        _assert_user_scope(search_b, USER_B)

        print(
            "[PASS] Mem0 memory isolation verified.",
            flush=True,
        )

    finally:
        # Clean up only the two dedicated test users.
        print(
            "[CLEANUP] Removing USER_A test memories",
            flush=True,
        )
        delete_user_memories(USER_A)

        print(
            "[CLEANUP] Removing USER_B test memories",
            flush=True,
        )
        delete_user_memories(USER_B)

        print(
            "[CLEANUP] Finished",
            flush=True,
        )
