from datetime import datetime
from typing import Callable

from module.chat_memory_module import (
    create_chat,
    get_recent_messages,
    get_user_chats,
    save_chat,
    save_message,
    update_chat,
)

from module.context_module import build_context

from module.conversation_model import (
    ConversationRequest,
    ConversationResponse,
)

from module.fact_extractor_module import extract_facts

from module.long_memory_module import get_user_facts

from module.message_model import Message

from module.rag_module import retrieve_context


def send_message(
    request: ConversationRequest,
    generate_answer: Callable[[str], str],
    use_rag: bool = True,
) -> ConversationResponse:
    """مدیریت کامل یک پیام در Conversation"""

    # =================================================
    # 1. Chat
    # =================================================

    chat_id: str = request.chat_id

    if not chat_id:
        chat = create_chat(
            user_id=request.user_id,
        )

        save_chat(
            chat=chat,
        )

        chat_id = chat.chat_id

    # =================================================
    # 2. Short Memory
    # =================================================

    messages: list[Message] = get_recent_messages(
        user_id=request.user_id,
        chat_id=chat_id,
        limit=5,
    )

    # =================================================
    # 3. Fact Extraction
    # =================================================

    extract_facts(
        user_id=request.user_id,
        message=request.message,
    )

    # =================================================
    # 4. Long Memory
    # =================================================

    facts = get_user_facts(
        user_id=request.user_id,
    )

    # =================================================
    # 5. RAG
    # =================================================

    rag_context: str = ""

    if use_rag:
        rag_context = retrieve_context(
            query=request.message,
        )

    # =================================================
    # 6. Context Builder
    # =================================================

    context: str = build_context(
        messages=messages,
        facts=facts,
        rag_context=rag_context,
        current_message=request.message,
    )

    # =================================================
    # 7. Generator
    # =================================================

    answer: str = generate_answer(
        context,
    )

    # =================================================
    # 8. Save User Message
    # =================================================

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

    # =================================================
    # 9. Save Assistant Message
    # =================================================

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

    # =================================================
    # 10. Update Chat
    # =================================================

    chats = get_user_chats(
        user_id=request.user_id,
    )

    current_chat = next(
        (
            chat
            for chat in chats
            if chat.chat_id == chat_id
        ),
        None,
    )

    if current_chat:
        current_chat.updated_at = datetime.now()

        update_chat(
            chat=current_chat,
        )

    # =================================================
    # 11. Response
    # =================================================

    return ConversationResponse(
        chat_id=chat_id,
        message=answer,
    )