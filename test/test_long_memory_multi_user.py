import os

from module.chat_memory_module import get_chat_messages
from module.conversation_model import ConversationRequest
from module.conversation_module import send_message
from module.generator_module import generate_answer
from module.long_memory_module import get_user_facts


CHAT_MEMORY_FILE_PATH = "data/chat_memory.json"
LONG_MEMORY_FILE_PATH = "data/long_memory.json"


def test_long_memory_multi_user() -> None:
    """تست Long Memory بین Chatهای مختلف و Userهای مختلف"""

    # -------------------------------------------------
    # Clear Memory
    # -------------------------------------------------

    if os.path.exists(CHAT_MEMORY_FILE_PATH):
        os.remove(CHAT_MEMORY_FILE_PATH)

    if os.path.exists(LONG_MEMORY_FILE_PATH):
        os.remove(LONG_MEMORY_FILE_PATH)

    # -------------------------------------------------
    # User 001 - Chat 001
    # -------------------------------------------------

    user_001 = "user_001"

    request_1 = ConversationRequest(
        chat_id="",
        user_id=user_001,
        message="اسم من علی است و برنامه‌نویس هستم.",
    )

    response_1 = send_message(
        request=request_1,
        generate_answer=generate_answer,
    )

    assert response_1.chat_id
    assert response_1.message

    chat_001: str = response_1.chat_id

    print()
    print("===== USER 001 - CHAT 001 =====")
    print(f"Chat ID: {chat_001}")
    print(f"User: {request_1.message}")
    print(f"Assistant: {response_1.message}")

    # -------------------------------------------------
    # User 001 - Chat 002
    # -------------------------------------------------

    request_2 = ConversationRequest(
        chat_id="",
        user_id=user_001,
        message="اسم من چیست؟",
    )

    response_2 = send_message(
        request=request_2,
        generate_answer=generate_answer,
    )

    assert response_2.chat_id
    assert response_2.chat_id != chat_001
    assert response_2.message

    chat_002: str = response_2.chat_id

    print()
    print("===== USER 001 - CHAT 002 =====")
    print(f"Chat ID: {chat_002}")
    print(f"User: {request_2.message}")
    print(f"Assistant: {response_2.message}")

    # -------------------------------------------------
    # Verify User 001 Facts
    # -------------------------------------------------

    user_001_facts = get_user_facts(
        user_id=user_001,
    )

    user_001_fact_values = {
        fact.key: fact.value
        for fact in user_001_facts
    }

    assert user_001_fact_values["name"] == "علی"
    assert user_001_fact_values["job"] == "برنامه‌نویس"

    # -------------------------------------------------
    # User 002 - New User
    # -------------------------------------------------

    user_002 = "user_002"

    request_3 = ConversationRequest(
        chat_id="",
        user_id=user_002,
        message="اسم من رضا است و حسابدار هستم.",
    )

    response_3 = send_message(
        request=request_3,
        generate_answer=generate_answer,
    )

    assert response_3.chat_id
    assert response_3.message

    chat_003: str = response_3.chat_id

    print()
    print("===== USER 002 - CHAT 001 =====")
    print(f"Chat ID: {chat_003}")
    print(f"User: {request_3.message}")
    print(f"Assistant: {response_3.message}")

    # -------------------------------------------------
    # Verify User 002 Facts
    # -------------------------------------------------

    user_002_facts = get_user_facts(
        user_id=user_002,
    )

    user_002_fact_values = {
        fact.key: fact.value
        for fact in user_002_facts
    }

    assert user_002_fact_values["name"] == "رضا"
    assert user_002_fact_values["job"] == "حسابدار"

    # -------------------------------------------------
    # Verify User Isolation
    # -------------------------------------------------

    assert user_001_fact_values["name"] != user_002_fact_values["name"]
    assert user_001_fact_values["job"] != user_002_fact_values["job"]

    # -------------------------------------------------
    # Verify Short Memory Isolation
    # -------------------------------------------------

    user_001_chat_002_messages = get_chat_messages(
        user_id=user_001,
        chat_id=chat_002,
    )

    user_002_chat_001_messages = get_chat_messages(
        user_id=user_002,
        chat_id=chat_003,
    )

    assert len(user_001_chat_002_messages) == 2
    assert len(user_002_chat_001_messages) == 2

    assert (
        user_001_chat_002_messages[0].content
        == "اسم من چیست؟"
    )

    assert (
        user_002_chat_001_messages[0].content
        == "اسم من رضا است و حسابدار هستم."
    )

    print()
    print("===== USER ISOLATION =====")
    print("User 001 facts:")

    for fact in user_001_facts:
        print(f"{fact.key}: {fact.value}")

    print()
    print("User 002 facts:")

    for fact in user_002_facts:
        print(f"{fact.key}: {fact.value}")

    print()
    print("test_long_memory_multi_user: PASSED")


if __name__ == "__main__":
    test_long_memory_multi_user()