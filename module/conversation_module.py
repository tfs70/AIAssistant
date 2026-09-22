from datetime import datetime
from typing import Callable

from module.chat_memory_module import (
    create_chat,
    get_recent_messages,
    save_chat,
    save_message,
)
from module.context_module import build_context
from module.conversation_model import (
    ConversationRequest,
    ConversationResponse,
)
from module.fact_extractor_module import extract_facts
from module.long_memory_module import (
    get_user_facts,
    save_fact,
)
from module.message_model import Message


def send_message(
    request: ConversationRequest,
    generate_answer: Callable[[str], str],
) -> ConversationResponse:
    """مدیریت یک پیام در Conversation"""

    # -------------------------------------------------
    # Chat
    # -------------------------------------------------

    chat_id: str = request.chat_id

    if not chat_id:
        chat = create_chat(
            user_id=request.user_id,
        )

        save_chat(
            chat=chat,
        )

        chat_id = chat.chat_id

    # -------------------------------------------------
    # Fact Extractor
    # -------------------------------------------------

    facts = extract_facts(
        user_id=request.user_id,
        message=request.message,
    )

    for fact in facts:
        save_fact(
            fact=fact,
        )

    # -------------------------------------------------
    # Short Memory
    # -------------------------------------------------

    messages: list[Message] = get_recent_messages(
        user_id=request.user_id,
        chat_id=chat_id,
        limit=5,
    )

    # -------------------------------------------------
    # Long Memory
    # -------------------------------------------------

    facts = get_user_facts(
        user_id=request.user_id,
    )

    # -------------------------------------------------
    # Context
    # -------------------------------------------------

    context: str = build_context(
        messages=messages,
        facts=facts,
        current_message=request.message,
    )

    # -------------------------------------------------
    # Generator
    # -------------------------------------------------

    answer: str = generate_answer(
        context,
    )

    # -------------------------------------------------
    # Save User Message
    # -------------------------------------------------

    user_message = Message(
        user_id=request.user_id,
        chat_id=chat_id,
        role="user",
        content=request.message,
        created_at=datetime.now(),
    )

    save_message(
        message=user_message,
    )

    # -------------------------------------------------
    # Save Assistant Message
    # -------------------------------------------------

    assistant_message = Message(
        user_id=request.user_id,
        chat_id=chat_id,
        role="assistant",
        content=answer,
        created_at=datetime.now(),
    )

    save_message(
        message=assistant_message,
    )

    # -------------------------------------------------
    # Response
    # -------------------------------------------------

    return ConversationResponse(
        chat_id=chat_id,
        message=answer,
    )