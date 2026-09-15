import os

from module.chat_memory_module import get_chat_messages
from module.conversation_model import ConversationRequest
from module.conversation_module import send_message
from module.generator_module import generate_answer


MEMORY_FILE_PATH = "data/chat_memory.json"


def test_real_conversation() -> None:
    """تست واقعی کل جریان Conversation"""

    if os.path.exists(MEMORY_FILE_PATH):
        os.remove(MEMORY_FILE_PATH)

    request = ConversationRequest(
        chat_id="",
        user_id="user_001",
        message="سلام، خودت را در یک جمله معرفی کن.",
    )

    response = send_message(
        request=request,
        generate_answer=generate_answer,
    )

    assert response.chat_id
    assert response.message

    print()
    print("Chat ID:")
    print(response.chat_id)

    print()
    print("Assistant:")
    print(response.message)

    messages = get_chat_messages(
        user_id="user_001",
        chat_id=response.chat_id,
    )

    assert len(messages) == 2

    assert messages[0].role == "user"
    assert messages[0].content == request.message

    assert messages[1].role == "assistant"
    assert messages[1].content == response.message

    print()
    print("Saved messages:")

    for message in messages:
        print(f"{message.role}: {message.content}")

    print()
    print("test_real_conversation: PASSED")


if __name__ == "__main__":
    test_real_conversation()

