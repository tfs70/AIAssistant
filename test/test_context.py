from datetime import datetime

from module.context_module import build_context
from module.message_model import Message
from module.user_fact_model import UserFact


def test_build_context() -> None:
    """تست ساخت Context"""

    now: datetime = datetime.now()

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

    context: str = build_context(
        messages=messages,
        facts=facts,
        current_message="اسم من چیست؟",
    )

    print()
    print("===== CONTEXT =====")
    print(context)
    print("===================")

    assert "user facts:" in context

    assert "name: علی" in context
    assert "job: برنامه‌نویس" in context

    assert "conversation:" in context
    assert "user: سلام." in context
    assert "assistant: سلام علی." in context

    assert "user: اسم من چیست؟" in context


if __name__ == "__main__":
    test_build_context()