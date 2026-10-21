
from unittest.mock import Mock

import pytest

from module import rag_module


def test_retrieve_context_returns_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test retrieval formatting without accessing the real Chroma index."""

    documents = [
        Mock(
            page_content="اطلاعات مربوط به سفارش خرید",
            metadata={
                "relative_source": "PurchaseOrder.pdf",
                "page_number": 2,
            },
        ),
        Mock(
            page_content="اطلاعات مربوط به تأمین‌کننده",
            metadata={
                "relative_source": "SupplyContract.pdf",
            },
        ),
    ]

    vector_store = Mock()
    vector_store.similarity_search.return_value = documents

    create_vector_store = Mock(return_value=vector_store)
    monkeypatch.setattr(
        rag_module,
        "create_vector_store",
        create_vector_store,
    )

    result = rag_module.retrieve_context(
        query="اطلاعات سفارش خرید",
        k=2,
    )

    create_vector_store.assert_called_once()
    vector_store.similarity_search.assert_called_once_with(
        query="اطلاعات سفارش خرید",
        k=2,
    )

    assert "اطلاعات مربوط به سفارش خرید" in result
    assert "اطلاعات مربوط به تأمین‌کننده" in result
    assert "PurchaseOrder.pdf" in result
    assert "SupplyContract.pdf" in result
    assert "page: 2" in result


def test_retrieve_context_returns_empty_for_blank_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank query must not open the vector store."""

    create_vector_store = Mock()
    monkeypatch.setattr(
        rag_module,
        "create_vector_store",
        create_vector_store,
    )

    result = rag_module.retrieve_context(query="   ")

    assert result == ""
    create_vector_store.assert_not_called()
