from datetime import datetime

from module.context_module import build_context
from module.message_model import Message


def test_build_context() -> None:
    """تست ساخت Context"""

    messages: list[Message] = [
        Message(
            user_id="user_001",
            chat_id="chat_001",
            role="user",
            content="اسم من علی است.",
            created_at=datetime.now(),
        ),
        Message(
            user_id="user_001",
            chat_id="chat_001",
            role="assistant",
            content="سلام علی.",
            created_at=datetime.now(),
        ),
    ]

    context: str = build_context(
        messages=messages,
        current_message="اسم من چیست؟",
    )

    print()
    print("===== CONTEXT =====")
    print(context)
    print("===================")

    assert "user: اسم من علی است." in context
    assert "assistant: سلام علی." in context
    assert "user: اسم من چیست؟" in context


if __name__ == "__main__":
    test_build_context()