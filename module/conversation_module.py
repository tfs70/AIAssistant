
import re
import time
from datetime import datetime
from typing import Callable, Optional

from module.chat_memory_module import (
    create_chat,
    get_recent_messages,
    get_user_chats,
    save_chat,
    save_message,
    update_chat,
)
from module.context_module import build_context
from module.conversation_model import ConversationRequest, ConversationResponse
from module.fact_extractor_module import extract_facts
from module.long_memory_module import get_user_facts, save_fact
from module.message_model import Message
from module.rag_module import retrieve_context


ToolExecutor = Callable[[str], Optional[str]]


def _may_contain_personal_fact(message: str) -> bool:
    """تشخیص اولیه پیام‌هایی که ممکن است اطلاعات شخصی جدید داشته باشند."""

    text = message.strip()

    personal_markers = (
        "اسم من",
        "نام من",
        "شغل من",
        "کار من",
        "من اسمم",
        "من نامم",
        "من اهل",
        "من متولد",
        "من کار می‌کنم",
        "من کار میکنم",
        "من هستم",
        "من برنامه‌نویس",
        "من برنامه نویس",
        "من متولد",
        "من در"
    )

    if any(marker in text for marker in personal_markers):
        return True

    # مثال: «من علی ام» یا «من علی‌ام»
    if re.search(r"من\s+\S+\s+(?:ام|هستم)", text):
        return True

    if re.search(r"من\s+\S+‌ام", text):
        return True

    return False


def send_message(
    request: ConversationRequest,
    generate_answer: Callable[[str], str],
    use_rag: bool = True,
    tool_executor: Optional[ToolExecutor] = None,
) -> ConversationResponse:
    """پردازش پیام با حافظه، RAG و ابزار اختیاری ERP."""

    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    if not request.user_id:
        raise ValueError("user_id is required.")

    if not request.message or not request.message.strip():
        raise ValueError("message cannot be empty.")

    user_id = request.user_id
    user_text = request.message.strip()
    chat_id = request.chat_id

    if not chat_id:
        chat = create_chat(user_id=user_id)
        save_chat(chat=chat)
        chat_id = chat.chat_id

    # 1. استخراج و ذخیره اطلاعات شخصی، فقط در صورت نیاز
    stage_started = time.perf_counter()

    if _may_contain_personal_fact(user_text):
        extracted_facts = extract_facts(
            user_id=user_id,
            message=user_text,
        )

        for fact in extracted_facts:
            save_fact(fact=fact)

    timings["fact_extraction_and_save"] = (
        time.perf_counter() - stage_started
    )

    # 2. دریافت حافظه کوتاه‌مدت و بلندمدت
    stage_started = time.perf_counter()

    messages = get_recent_messages(
        user_id=user_id,
        chat_id=chat_id,
        limit=5,
    )

    facts = get_user_facts(user_id=user_id)

    timings["memory_retrieval"] = time.perf_counter() - stage_started

    # 3. بازیابی اسناد RAG
    stage_started = time.perf_counter()

    rag_context = ""

    if use_rag:
        rag_context = retrieve_context(query=user_text)

    timings["rag_retrieval"] = time.perf_counter() - stage_started

    # 4. ساخت Context
    context = build_context(
        messages=messages,
        facts=facts,
        rag_context=rag_context,
        current_message=user_text,
    )

    # 5. تصمیم‌گیری و اجرای ابزار ERP در صورت نیاز
    stage_started = time.perf_counter()

    tool_result = None

    if tool_executor is not None:
        tool_result = tool_executor(user_text)

    timings["tool_routing_and_execution"] = (
        time.perf_counter() - stage_started
    )

    final_context = context

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

    # 6. تولید پاسخ
    stage_started = time.perf_counter()

    answer = generate_answer(final_context)

    timings["answer_generation"] = time.perf_counter() - stage_started

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Generator returned an empty answer.")

    answer = answer.strip()

    total_seconds = time.perf_counter() - total_started

    # ثبت زمان واقعی در متن ذخیره‌شده، تا در تاریخچه هم دیده شود.
    answer_with_timing = (
        f"{answer}\n\n"
        f"---\n"
        f"⏱️ زمان پاسخ: {total_seconds:.2f} ثانیه"
    )

    # 7. ذخیره پیام‌ها
    now = datetime.now()

    user_message = Message(
        user_id=user_id,
        chat_id=chat_id,
        role="user",
        content=user_text,
        created_at=now,
    )
    save_message(message=user_message)

    assistant_message = Message(
        user_id=user_id,
        chat_id=chat_id,
        role="assistant",
        content=answer_with_timing,
        created_at=datetime.now(),
    )
    save_message(message=assistant_message)

    chats = get_user_chats(user_id=user_id)

    current_chat = next(
        (chat for chat in chats if chat.chat_id == chat_id),
        None,
    )

    if current_chat is not None:
        current_chat.updated_at = datetime.now()
        update_chat(chat=current_chat)

    # 8. لاگ زمان مراحل
    print(
        "\n===== CONVERSATION TIMINGS =====\n"
        f"Fact extraction/save: "
        f"{timings['fact_extraction_and_save']:.2f}s\n"
        f"Memory retrieval: "
        f"{timings['memory_retrieval']:.2f}s\n"
        f"RAG retrieval: "
        f"{timings['rag_retrieval']:.2f}s\n"
        f"Tool routing/execution: "
        f"{timings['tool_routing_and_execution']:.2f}s\n"
        f"Answer generation: "
        f"{timings['answer_generation']:.2f}s\n"
        f"Total: {total_seconds:.2f}s\n"
    )

    return ConversationResponse(
        chat_id=chat_id,
        message=answer_with_timing,
    )
