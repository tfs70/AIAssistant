import os

from module.chat_memory_module import get_chat_messages
from module.conversation_model import ConversationRequest
from module.conversation_module import send_message
from module.generator_module import generate_answer


MEMORY_FILE_PATH = "data/chat_memory.json"


def test_real_conversation() -> None:
    """تست واقعی چند پیام متوالی در یک Conversation"""

    if os.path.exists(MEMORY_FILE_PATH):
        os.remove(MEMORY_FILE_PATH)

    user_id: str = "user_001"

    # -------------------------------------------------
    # Message 1
    # -------------------------------------------------

    request_1 = ConversationRequest(
        chat_id="",
        user_id=user_id,
        message="اسم من علی است و برنامه‌نویس هستم.",
    )

    response_1 = send_message(
        request=request_1,
        generate_answer=generate_answer,
    )

    assert response_1.chat_id
    assert response_1.message

    chat_id: str = response_1.chat_id

    print()
    print("Message 1")
    print(f"User: {request_1.message}")
    print(f"Assistant: {response_1.message}")

    # -------------------------------------------------
    # Message 2
    # -------------------------------------------------

    request_2 = ConversationRequest(
        chat_id=chat_id,
        user_id=user_id,
        message="اسم من چیست؟",
    )

    response_2 = send_message(
        request=request_2,
        generate_answer=generate_answer,
    )

    assert response_2.chat_id == chat_id
    assert response_2.message

    print()
    print("Message 2")
    print(f"User: {request_2.message}")
    print(f"Assistant: {response_2.message}")

    # -------------------------------------------------
    # Message 3
    # -------------------------------------------------

    request_3 = ConversationRequest(
        chat_id=chat_id,
        user_id=user_id,
        message="شغل من چیست؟",
    )

    response_3 = send_message(
        request=request_3,
        generate_answer=generate_answer,
    )

    assert response_3.chat_id == chat_id
    assert response_3.message

    print()
    print("Message 3")
    print(f"User: {request_3.message}")
    print(f"Assistant: {response_3.message}")

    # -------------------------------------------------
    # Verify Chat Memory
    # -------------------------------------------------

    messages = get_chat_messages(
        user_id=user_id,
        chat_id=chat_id,
    )

    assert len(messages) == 6

    assert messages[0].role == "user"
    assert messages[0].content == request_1.message

    assert messages[1].role == "assistant"
    assert messages[1].content == response_1.message

    assert messages[2].role == "user"
    assert messages[2].content == request_2.message

    assert messages[3].role == "assistant"
    assert messages[3].content == response_2.message

    assert messages[4].role == "user"
    assert messages[4].content == request_3.message

    assert messages[5].role == "assistant"
    assert messages[5].content == response_3.message

    print()
    print("Saved messages:")

    for message in messages:
        print(
            f"{message.role}: {message.content}",
        )

    print()
    print("test_real_conversation: PASSED")


if __name__ == "__main__":
    test_real_conversation()

