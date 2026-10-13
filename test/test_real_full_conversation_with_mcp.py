import os

from module.chat_memory_module import get_chat_messages
from module.conversation_model import ConversationRequest
from module.conversation_module import send_message
from module.generator_module import generate_answer
from module.long_memory_module import get_user_facts


CHAT_MEMORY_FILE_PATH = "data/chat_memory.json"
LONG_MEMORY_FILE_PATH = "data/long_memory.json"


def test_real_full_conversation_with_mcp() -> None:
    """تست کامل Conversation با Memory و RAG"""

    # =================================================
    # Clean
    # =================================================

    if os.path.exists(
        CHAT_MEMORY_FILE_PATH,
    ):
        os.remove(
            CHAT_MEMORY_FILE_PATH,
        )

    if os.path.exists(
        LONG_MEMORY_FILE_PATH,
    ):
        os.remove(
            LONG_MEMORY_FILE_PATH,
        )

    # =================================================
    # User
    # =================================================

    user_id: str = "user_001"

    # =================================================
    # Message 1
    # =================================================

    request_1 = ConversationRequest(
        chat_id="",
        user_id=user_id,
        message="اسم من علی است و برنامه‌نویس هستم.",
    )

    response_1 = send_message(
        request=request_1,
        generate_answer=generate_answer,
        use_rag=True,
    )

    assert response_1.chat_id
    assert response_1.message

    chat_id: str = response_1.chat_id

    print()
    print("===== MESSAGE 1 =====")
    print(
        f"User: {request_1.message}",
    )
    print(
        f"Assistant: {response_1.message}",
    )
    print(
        f"Chat ID: {chat_id}",
    )

    # =================================================
    # Long Memory
    # =================================================

    facts = get_user_facts(
        user_id=user_id,
    )

    print()
    print("===== LONG MEMORY =====")

    for fact in facts:
        print(
            f"{fact.key}: {fact.value}",
        )

    fact_values = {
        fact.key: fact.value
        for fact in facts
    }

    assert fact_values["name"] == "علی"
    assert fact_values["job"] == "برنامه‌نویس"

    # =================================================
    # Message 2
    # =================================================

    request_2 = ConversationRequest(
        chat_id=chat_id,
        user_id=user_id,
        message="اسم من چیست؟",
    )

    response_2 = send_message(
        request=request_2,
        generate_answer=generate_answer,
        use_rag=True,
    )

    assert response_2.chat_id == chat_id
    assert response_2.message

    print()
    print("===== MESSAGE 2 =====")
    print(
        f"User: {request_2.message}",
    )
    print(
        f"Assistant: {response_2.message}",
    )

    # =================================================
    # Message 3 - RAG
    # =================================================

    request_3 = ConversationRequest(
        chat_id=chat_id,
        user_id=user_id,
        message="فروش شرکت در شهریور ۱۴۰۵ چقدر بوده است؟",
    )

    response_3 = send_message(
        request=request_3,
        generate_answer=generate_answer,
        use_rag=True,
    )

    assert response_3.chat_id == chat_id
    assert response_3.message

    print()
    print("===== MESSAGE 3 =====")
    print(
        f"User: {request_3.message}",
    )
    print(
        f"Assistant: {response_3.message}",
    )

    # =================================================
    # Message 4 - RAG + Short Memory
    # =================================================

    request_4 = ConversationRequest(
        chat_id=chat_id,
        user_id=user_id,
        message="میزان فروش نسبت به مرداد چقدر بیشتر بوده است؟",
    )

    response_4 = send_message(
        request=request_4,
        generate_answer=generate_answer,
        use_rag=True,
    )

    assert response_4.chat_id == chat_id
    assert response_4.message

    print()
    print("===== MESSAGE 4 =====")
    print(
        f"User: {request_4.message}",
    )
    print(
        f"Assistant: {response_4.message}",
    )

    # =================================================
    # Saved Messages
    # =================================================

    messages = get_chat_messages(
        user_id=user_id,
        chat_id=chat_id,
    )

    assert len(messages) == 8

    print()
    print("===== SAVED MESSAGES =====")

    for message in messages:
        print(
            f"{message.role}: {message.content}",
        )

    # =================================================
    # Message Verification
    # =================================================

    assert messages[0].role == "user"
    assert messages[0].content == request_1.message

    assert messages[1].role == "assistant"

    assert messages[2].role == "user"
    assert messages[2].content == request_2.message

    assert messages[3].role == "assistant"

    assert messages[4].role == "user"
    assert messages[4].content == request_3.message

    assert messages[5].role == "assistant"

    assert messages[6].role == "user"
    assert messages[6].content == request_4.message

    assert messages[7].role == "assistant"

    print()
    print(
        "test_real_full_conversation_with_mcp: PASSED",
    )


if __name__ == "__main__":
    test_real_full_conversation_with_mcp()