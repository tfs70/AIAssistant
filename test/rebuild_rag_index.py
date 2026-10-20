
from module.rag_module import rebuild_vector_store


def main() -> None:
    result = rebuild_vector_store()

    print("\n===== FINAL SUMMARY =====")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
