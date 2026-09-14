import os

from module.chat_memory_module import (
    get_chat_messages,
    get_user_chats,
)
from module.conversation_model import ConversationRequest
from module.conversation_module import send_message


MEMORY_FILE_PATH = "data/chat_memory.json"


def fake_generate_answer(
    query: str,
) -> str:
    """Generator تستی"""

    return f"Fake answer: {query}"


def test_new_chat() -> None:
    """تست ایجاد Chat جدید"""

    if os.path.exists(MEMORY_FILE_PATH):
        os.remove(MEMORY_FILE_PATH)

    request = ConversationRequest(
        chat_id="",
        user_id="user_001",
        message="سلام",
    )

    response = send_message(
        request=request,
        generate_answer=fake_generate_answer,
    )

    assert response.chat_id
    assert response.message

    chats = get_user_chats(
        user_id="user_001",
    )

    assert len(chats) == 1
    assert chats[0].chat_id == response.chat_id

    messages = get_chat_messages(
        user_id="user_001",
        chat_id=response.chat_id,
    )

    assert len(messages) == 2

    assert messages[0].role == "user"
    assert messages[0].content == "سلام"

    assert messages[1].role == "assistant"
    assert messages[1].content == response.message

    print("test_new_chat: PASSED")


def test_continue_chat() -> None:
    """تست ادامه Chat موجود"""

    chats = get_user_chats(
        user_id="user_001",
    )

    assert len(chats) == 1

    chat_id: str = chats[0].chat_id

    request = ConversationRequest(
        chat_id=chat_id,
        user_id="user_001",
        message="حالت چطوره؟",
    )

    response = send_message(
        request=request,
        generate_answer=fake_generate_answer,
    )

    assert response.chat_id == chat_id
    assert response.message

    messages = get_chat_messages(
        user_id="user_001",
        chat_id=chat_id,
    )

    assert len(messages) == 4

    assert messages[0].role == "user"
    assert messages[0].content == "سلام"

    assert messages[1].role == "assistant"

    assert messages[2].role == "user"
    assert messages[2].content == "حالت چطوره؟"

    assert messages[3].role == "assistant"
    assert messages[3].content == response.message

    print("test_continue_chat: PASSED")


def test_multiple_users_and_chats() -> None:
    """تست جداسازی کاربران و Chatها"""

    request_user_002 = ConversationRequest(
        chat_id="",
        user_id="user_002",
        message="سلام از کاربر دوم",
    )

    response_user_002 = send_message(
        request=request_user_002,
        generate_answer=fake_generate_answer,
    )

    request_user_001_new_chat = ConversationRequest(
        chat_id="",
        user_id="user_001",
        message="این یک گفتگوی جدید است",
    )

    response_user_001_new_chat = send_message(
        request=request_user_001_new_chat,
        generate_answer=fake_generate_answer,
    )

    user_001_chats = get_user_chats(
        user_id="user_001",
    )

    user_002_chats = get_user_chats(
        user_id="user_002",
    )

    assert len(user_001_chats) == 2
    assert len(user_002_chats) == 1

    user_001_chat_ids = [
        chat.chat_id
        for chat in user_001_chats
    ]

    user_002_chat_ids = [
        chat.chat_id
        for chat in user_002_chats
    ]

    assert response_user_002.chat_id in user_002_chat_ids
    assert response_user_002.chat_id not in user_001_chat_ids

    assert response_user_001_new_chat.chat_id in user_001_chat_ids
    assert response_user_001_new_chat.chat_id != response_user_002.chat_id

    print("test_multiple_users_and_chats: PASSED")


if __name__ == "__main__":
    test_new_chat()
    test_continue_chat()
    test_multiple_users_and_chats()

    print()
    print("All conversation tests passed.")
