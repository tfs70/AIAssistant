from datetime import datetime

from langchain_core.documents import Document

from module.context_module import build_context
from module.message_model import Message
from module.user_fact_model import UserFact


def test_context_with_rag() -> None:
    """Test Context Builder with Short Memory, Long Memory and RAG"""

    # -------------------------------------------------
    # Long Memory
    # -------------------------------------------------

    now: datetime = datetime.now()

    facts: list[UserFact] = [
        UserFact(
            user_id="user_001",
            key="name",
            value="علی",
            created_at=now,
            updated_at=now,
        ),
        UserFact(
            user_id="user_001",
            key="job",
            value="برنامه‌نویس",
            created_at=now,
            updated_at=now,
        ),
    ]

    # -------------------------------------------------
    # Short Memory
    # -------------------------------------------------

    messages: list[Message] = [
        Message(
            user_id="user_001",
            chat_id="chat_001",
            role="user",
            content="سلام.",
            created_at=now,
        ),
        Message(
            user_id="user_001",
            chat_id="chat_001",
            role="assistant",
            content="سلام علی.",
            created_at=now,
        ),
    ]

    # -------------------------------------------------
    # RAG
    # -------------------------------------------------

    rag_documents: list[Document] = [
        Document(
            page_content=(
                "فروش شرکت در ماه شهریور ۱۴۰۵ "
                "برابر با ۱۲ میلیارد تومان بوده است."
            ),
        ),
        Document(
            page_content=(
                "تعداد مشتریان فعال شرکت در شهریور ۱۴۰۵ "
                "برابر با ۸۵۰ مشتری بوده است."
            ),
        ),
    ]

    # -------------------------------------------------
    # Current Message
    # -------------------------------------------------

    current_message: str = (
        "فروش شرکت در شهریور ۱۴۰۵ چقدر بوده است؟"
    )

    # -------------------------------------------------
    # Build Context
    # -------------------------------------------------

    context: str = build_context(
        messages=messages,
        facts=facts,
        rag_documents=rag_documents,
        current_message=current_message,
    )

    print()
    print("===== CONTEXT =====")
    print(context)
    print("===================")

    # -------------------------------------------------
    # Verify Long Memory
    # -------------------------------------------------

    assert "user facts:" in context
    assert "name: علی" in context
    assert "job: برنامه‌نویس" in context

    # -------------------------------------------------
    # Verify Short Memory
    # -------------------------------------------------

    assert "conversation:" in context
    assert "user: سلام." in context
    assert "assistant: سلام علی." in context

    # -------------------------------------------------
    # Verify RAG
    # -------------------------------------------------

    assert "relevant documents:" in context
    assert "۱۲ میلیارد تومان" in context
    assert "۸۵۰ مشتری" in context

    # -------------------------------------------------
    # Verify Current Message
    # -------------------------------------------------

    assert (
        "user: فروش شرکت در شهریور ۱۴۰۵ چقدر بوده است؟"
        in context
    )

    print()
    print("test_context_with_rag: PASSED")


if __name__ == "__main__":
    test_context_with_rag()