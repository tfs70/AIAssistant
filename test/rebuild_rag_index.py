
import argparse

from module.rag_module import rebuild_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild or incrementally synchronize the RAG index."
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Index new/changed files without clearing the existing index.",
    )

    args = parser.parse_args()

    result = rebuild_vector_store(
        clear_existing=not args.incremental,
    )

    print("\n===== FINAL SUMMARY =====")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
