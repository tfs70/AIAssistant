
from module.rag_module import (
    create_vector_store,
    retrieve_context,
    sync_rag_documents,
)


def main() -> None:
    print("\n===== STEP 1: SYNC =====")
    print(sync_rag_documents())

    print("\n===== STEP 2: COLLECTION =====")
    vector_store = create_vector_store()

    print("Collection count:", vector_store._collection.count())

    query = "رسید انبار چیست؟"

    print("\n===== STEP 3: DIRECT SEARCH =====")
    results = vector_store.similarity_search(query=query, k=5)

    print("Retrieved documents:", len(results))

    for index, document in enumerate(results, start=1):
        print(f"\n--- Document {index} ---")
        print("Source:", document.metadata.get("source"))
        print(document.page_content)

    print("\n===== STEP 4: RETRIEVE CONTEXT =====")
    context = retrieve_context(query=query)

    print("Context length:", len(context))
    print("Context:")
    print(context if context else "[EMPTY CONTEXT]")


if __name__ == "__main__":
    main()
