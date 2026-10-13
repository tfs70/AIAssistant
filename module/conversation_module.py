
from datetime import datetime
from typing import Callable, Any, Optional

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


ToolExecutor = Callable[[str], Optional[str]]


def send_message(
    request: ConversationRequest,
    generate_answer: Callable[[str], str],
    use_rag: bool = True,
    tool_executor: Optional[ToolExecutor] = None,
) -> ConversationResponse:
    """مدیریت یک پیام در Conversation."""

    # =================================================
    # Validate Request
    # =================================================

    if not request.user_id:
        raise ValueError("user_id is required.")

    if not request.message or not request.message.strip():
        raise ValueError("message cannot be empty.")

    user_id: str = request.user_id
    user_text: str = request.message.strip()

    # =================================================
    # Chat
    # =================================================

    chat_id: str = request.chat_id

    if not chat_id:
        chat = create_chat(
            user_id=user_id,
        )

        save_chat(
            chat=chat,
        )

        chat_id = chat.chat_id

    # =================================================
    # Short Memory
    # =================================================

    messages: list[Message] = get_recent_messages(
        user_id=user_id,
        chat_id=chat_id,
        limit=5,
    )

    # =================================================
    # Fact Extraction
    # =================================================

    # حفظ حقایق باید در خود extract_facts انجام شود.
    # اگر این تابع فقط حقایق استخراج‌شده را برگرداند،
    # باید آن‌ها را طبق API واقعی long_memory_module ذخیره کنیم.
    extract_facts(
        user_id=user_id,
        message=user_text,
    )

    # =================================================
    # Long Memory
    # =================================================

    facts = get_user_facts(
        user_id=user_id,
    )

    # =================================================
    # RAG
    # =================================================

    rag_context: str = ""

    if use_rag:
        rag_context = retrieve_context(
            query=user_text,
        )

    # =================================================
    # Base Context
    # =================================================

    context: str = build_context(
        messages=messages,
        facts=facts,
        rag_context=rag_context,
        current_message=user_text,
    )

    # =================================================
    # Optional MCP Tool
    # =================================================

    tool_result: Optional[str] = None

    if tool_executor is not None:
        # قرارداد این تابع:
        # - ابزار لازم نیست: None برگرداند.
        # - ابزار اجرا شد: نتیجه را به شکل str برگرداند.
        # - خطا رخ داد: exception را پنهان نکند.
        tool_result = tool_executor(
            user_text,
        )

    # =================================================
    # Final Context
    # =================================================

    final_context: str = context

    if tool_result is not None:
        final_context += (
            "\n\n===== TRUSTED ERP TOOL RESULT =====\n"
            f"{tool_result}\n\n"
            "قوانین استفاده از نتیجه ERP:\n"
            "- پاسخ را بر اساس نتیجه واقعی ابزار بنویس.\n"
            "- نام‌ها، اعداد و مقادیر نتیجه را تغییر نده.\n"
            "- داده‌ای را که ابزار برنگردانده است، نساز.\n"
            "- اگر نتیجه برای پاسخ کافی نیست، این موضوع را بیان کن.\n"
            "- پاسخ نهایی باید فارسی باشد."
        )

    # =================================================
    # Generator
    # =================================================

    answer: str = generate_answer(
        final_context,
    )

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Generator returned an empty answer.")

    answer = answer.strip()

    # =================================================
    # Save User Message
    # =================================================

    user_message = Message(
        user_id=user_id,
        chat_id=chat_id,
        role="user",
        content=user_text,
        created_at=datetime.now(),
    )

    save_message(
        message=user_message,
    )

    # =================================================
    # Save Assistant Message
    # =================================================

    assistant_message = Message(
        user_id=user_id,
        chat_id=chat_id,
        role="assistant",
        content=answer,
        created_at=datetime.now(),
    )

    save_message(
        message=assistant_message,
    )

    # =================================================
    # Update Chat
    # =================================================

    chats = get_user_chats(
        user_id=user_id,
    )

    current_chat = next(
        (
            chat
            for chat in chats
            if chat.chat_id == chat_id
        ),
        None,
    )

    if current_chat is not None:
        current_chat.updated_at = datetime.now()

        update_chat(
            chat=current_chat,
        )

    # =================================================
    # Response
    # =================================================

    return ConversationResponse(
        chat_id=chat_id,
        message=answer,
    )
