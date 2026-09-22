import os
from datetime import datetime

from module.chat_memory_module import get_chat_messages
from module.conversation_model import ConversationRequest
from module.conversation_module import send_message
from module.generator_module import generate_answer
from module.long_memory_module import save_fact
from module.user_fact_model import UserFact


CHAT_MEMORY_FILE_PATH = "data/chat_memory.json"
LONG_MEMORY_FILE_PATH = "data/long_memory.json"


def test_real_long_memory_conversation() -> None:
    """تست واقعی Long Memory در Conversation"""

    # -------------------------------------------------
    # Clear Memory
    # -------------------------------------------------

    if os.path.exists(CHAT_MEMORY_FILE_PATH):
        os.remove(CHAT_MEMORY_FILE_PATH)

    if os.path.exists(LONG_MEMORY_FILE_PATH):
        os.remove(LONG_MEMORY_FILE_PATH)

    user_id: str = "user_001"

    now: datetime = datetime.now()

    # -------------------------------------------------
    # Save Long Memory
    # -------------------------------------------------

    name_fact = UserFact(
        user_id=user_id,
        key="name",
        value="علی",
        created_at=now,
        updated_at=now,
    )

    save_fact(
        fact=name_fact,
    )

    job_fact = UserFact(
        user_id=user_id,
        key="job",
        value="برنامه‌نویس",
        created_at=now,
        updated_at=now,
    )

    save_fact(
        fact=job_fact,
    )

    # -------------------------------------------------
    # New Chat
    # -------------------------------------------------

    request = ConversationRequest(
        chat_id="",
        user_id=user_id,
        message="اسم من چیست؟",
    )

    response = send_message(
        request=request,
        generate_answer=generate_answer,
    )

    assert response.chat_id
    assert response.message

    print()
    print("===== LONG MEMORY TEST =====")
    print(f"User: {request.message}")
    print(f"Assistant: {response.message}")
    print(f"Chat ID: {response.chat_id}")
    print("============================")

    # -------------------------------------------------
    # Verify Chat
    # -------------------------------------------------

    messages = get_chat_messages(
        user_id=user_id,
        chat_id=response.chat_id,
    )

    assert len(messages) == 2

    assert messages[0].role == "user"
    assert messages[0].content == "اسم من چیست؟"

    assert messages[1].role == "assistant"

    print()
    print("test_real_long_memory_conversation: PASSED")


if __name__ == "__main__":
    test_real_long_memory_conversation()