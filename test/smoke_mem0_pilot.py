
from module.mem0_memory_module import (
    add_memory,
    delete_user_memories,
    get_all_memories,
    search_memories,
)

USER_ID = "MEM0_PILOT_SMOKE_USER"


def main() -> None:
    print("=== Mem0 Pilot Diagnostic ===")
    print(f"User ID: {USER_ID}")

    # Clean up only this dedicated test user's data before the test.
    delete_user_memories(USER_ID)

    try:
        text = (
            "آزمایش Mem0 پروژه Persis: "
            "نام آزمایشی کاربر آذر است."
        )

        print("\n[1] Adding temporary memory...")
        add_result = add_memory(
            USER_ID,
            text,
            key="smoke_test",
        )
        print("Add result:")
        print(add_result)

        print("\n[2] Reading all memories for this user...")
        all_memories = get_all_memories(USER_ID)
        print(f"Memory count: {len(all_memories)}")
        for item in all_memories:
            print(item)

        if not all_memories:
            print(
                "\nDIAGNOSIS: No memory was returned by get_all_memories. "
                "Investigate add() and user_id filtering."
            )
            return

        print("\n[3] Searching for the same user's memory...")
        query = "نام آزمایشی کاربر چیست؟"
        results = search_memories(
            USER_ID,
            query,
            limit=5,
        )
        print(f"Search result count: {len(results)}")
        for item in results:
            print(item)

        if results:
            print("\nRESULT: Add, read, and semantic search returned data.")
        else:
            print(
                "\nDIAGNOSIS: Direct reading found a memory, "
                "but semantic search returned no results. "
                "Investigate search/filter behavior."
            )

    finally:
        print("\n[4] Cleaning up only the temporary test user's memories...")
        delete_user_memories(USER_ID)
        print("Temporary test-user memories deleted.")


if __name__ == "__main__":
    main()
