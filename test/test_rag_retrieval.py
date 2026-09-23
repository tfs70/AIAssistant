import os
import shutil

from module.rag_module import (
    get_or_build_vector_store,
    load_text_file,
    retrieve_context,
    split_documents,
)


TEST_FILE_PATH: str = "data/test_rag.txt"
CHROMA_DIRECTORY: str = "data/chroma_db"


def test_rag_retrieval() -> None:
    """Test RAG retrieval"""

    if os.path.exists(path=CHROMA_DIRECTORY):
        shutil.rmtree(
            path=CHROMA_DIRECTORY,
        )

    documents = load_text_file(
        file_path=TEST_FILE_PATH,
    )

    assert len(documents) > 0

    chunks = split_documents(
        documents=documents,
        chunk_size=200,
        chunk_overlap=20,
    )

    assert len(chunks) > 1

    print()
    print("===== DOCUMENTS =====")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    vector_store = get_or_build_vector_store(
        chunks=chunks,
    )

    # -------------------------------------------------
    # Sales
    # -------------------------------------------------

    sales_query: str = "فروش شرکت در شهریور ۱۴۰۵ چقدر بوده است؟"

    sales_results = retrieve_context(
        query=sales_query,
        vector_store=vector_store,
        k=2,
    )

    assert len(sales_results) > 0

    sales_context: str = "\n".join(
        result.page_content
        for result in sales_results
    )

    print()
    print("===== SALES RETRIEVAL =====")
    print(f"Query: {sales_query}")
    print(sales_context)

    assert "۱۲ میلیارد تومان" in sales_context

    # -------------------------------------------------
    # Inventory
    # -------------------------------------------------

    inventory_query: str = "موجودی انبار شرکت چقدر بوده است؟"

    inventory_results = retrieve_context(
        query=inventory_query,
        vector_store=vector_store,
        k=2,
    )

    assert len(inventory_results) > 0

    inventory_context: str = "\n".join(
        result.page_content
        for result in inventory_results
    )

    print()
    print("===== INVENTORY RETRIEVAL =====")
    print(f"Query: {inventory_query}")
    print(inventory_context)

    assert "۲ میلیارد تومان" in inventory_context

    # -------------------------------------------------
    # Customers
    # -------------------------------------------------

    customer_query: str = "تعداد مشتریان فعال شرکت چقدر بوده است؟"

    customer_results = retrieve_context(
        query=customer_query,
        vector_store=vector_store,
        k=2,
    )

    assert len(customer_results) > 0

    customer_context: str = "\n".join(
        result.page_content
        for result in customer_results
    )

    print()
    print("===== CUSTOMER RETRIEVAL =====")
    print(f"Query: {customer_query}")
    print(customer_context)

    assert "۸۵۰ مشتری" in customer_context

    print()
    print("test_rag_retrieval: PASSED")


if __name__ == "__main__":
    test_rag_retrieval()