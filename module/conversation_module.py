
import json
import re
import time
from datetime import datetime
from typing import Any, Callable, Optional

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
from module.mcp_tool_executor_module import route_and_execute_question
from module.rag_module import retrieve_context, sync_rag_documents


RouteExecutor = Callable[[str], dict[str, Any]]


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
        "من در",
    )

    if any(marker in text for marker in personal_markers):
        return True

    if re.search(r"من\s+\S+\s+(?:ام|هستم)", text):
        return True

    if re.search(r"من\s+\S+‌ام", text):
        return True

    return False


def _format_tool_results(tool_results: list[dict[str, Any]]) -> str:
    """تبدیل نتایج ساختاریافته ابزارها به متنی قابل استفاده برای مدل پاسخ‌گو."""

    if not tool_results:
        return ""

    return json.dumps(
        tool_results,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _build_trusted_data_context(
    route: str,
    tool_results: list[dict[str, Any]],
    rag_context: str,
) -> str:
    """ساخت بخش داده‌های معتبر، متناسب با مسیر تصمیم‌گیری."""

    sections: list[str] = []

    if tool_results:
        sections.append(
            "===== TRUSTED ERP TOOL RESULTS =====\n"
            f"{_format_tool_results(tool_results)}\n\n"
            "قوانین استفاده از نتایج ابزار:\n"
            "- داده‌ها را فقط بر اساس خروجی واقعی ابزار گزارش کن.\n"
            "- نام‌ها، شناسه‌ها، اعداد و مقادیر خروجی را تغییر نده.\n"
            "- اگر ابزار خطا داده است، خطا را به‌عنوان داده واقعی معرفی نکن.\n"
            "- اگر نتیجه کافی نیست، محدودیت را صریح بیان کن.\n"
            "- بین خرید واقعی، سفارش خرید و سایر مفاهیم تمایز بگذار."
        )

    if rag_context.strip():
        sections.append(
            "===== RETRIEVED DOCUMENT CONTEXT =====\n"
            f"{rag_context}\n\n"
            "قوانین استفاده از اسناد:\n"
            "- فقط اطلاعات مرتبط با سؤال را از اسناد استفاده کن.\n"
            "- اطلاعات نامرتبط را وارد پاسخ نکن.\n"
            "- محتوای بازیابی‌شده را با نتیجه ابزار ERP اشتباه نگیر.\n"
            "- اگر اسناد پاسخ را پشتیبانی نمی‌کنند، این موضوع را بیان کن."
        )

    sections.append(
        "===== RESPONSE RULES =====\n"
        f"Selected route: {route}\n"
        "- پاسخ نهایی باید فارسی و مستقیم باشد.\n"
        "- برای اطلاعاتی که در منابع موجود نیستند، چیزی نساز."
    )

    return "\n\n".join(sections)


def send_message(
    request: ConversationRequest,
    generate_answer: Callable[[str], str],
    use_rag: bool = True,
    tool_executor: Optional[RouteExecutor] = None,
) -> ConversationResponse:
    """پردازش پیام با حافظه، تصمیم‌گیری، MCP، RAG و تولید پاسخ."""

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

    # 1. استخراج و ذخیره اطلاعات شخصی
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

    # 3. تصمیم‌گیری و اجرای ابزارها
    stage_started = time.perf_counter()

    executor = tool_executor or route_and_execute_question
    routing_result = executor(user_text)

    if not isinstance(routing_result, dict):
        raise TypeError(
            "Route executor must return a dictionary containing "
            "'route', 'rag_query' and 'tool_results'. "
            "Pass route_and_execute_question, not execute_mcp_for_question."
        )

    route = routing_result.get("route")

    valid_routes = {"chat", "tool", "rag", "tool_and_rag"}
    if route not in valid_routes:
        raise ValueError(f"Invalid route returned by executor: {route!r}")

    tool_results = routing_result.get("tool_results", [])
    if not isinstance(tool_results, list):
        raise TypeError("tool_results must be a list.")

    timings["tool_routing_and_execution"] = (
        time.perf_counter() - stage_started
    )

    # 4. بازیابی RAG فقط زمانی که مسیر انتخابی به آن نیاز دارد
    stage_started = time.perf_counter()

    rag_context = ""
    rag_used = use_rag and route in {"rag", "tool_and_rag"}

    if rag_used:
        sync_result = sync_rag_documents()

        print(
            "\n===== RAG SYNC RESULT =====\n"
            f"Added: {sync_result['added']}\n"
            f"Updated: {sync_result['updated']}\n"
            f"Skipped: {sync_result['skipped']}\n"
            f"Deleted: {sync_result['deleted']}\n"
        )

        rag_query = routing_result.get("rag_query") or user_text
        rag_context = retrieve_context(query=rag_query)

    timings["rag_retrieval"] = time.perf_counter() - stage_started

    # 5. ساخت زمینه مکالمه
    context = build_context(
        messages=messages,
        facts=facts,
        rag_context=rag_context,
        current_message=user_text,
    )

    trusted_context = _build_trusted_data_context(
        route=route,
        tool_results=tool_results,
        rag_context=rag_context,
    )

    final_context = f"{context}\n\n{trusted_context}"

    # 6. تولید پاسخ
    stage_started = time.perf_counter()

    answer = generate_answer(final_context)

    timings["answer_generation"] = time.perf_counter() - stage_started

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Generator returned an empty answer.")

    answer = answer.strip()
    total_seconds = time.perf_counter() - total_started

    answer_with_timing = (
        f"{answer}\n\n"
        f"---\n"
        f"⏱️ زمان پاسخ: {total_seconds:.2f} ثانیه"
    )

    # 7. ذخیره پیام‌ها
    now = datetime.now()

    save_message(
        message=Message(
            user_id=user_id,
            chat_id=chat_id,
            role="user",
            content=user_text,
            created_at=now,
        )
    )

    save_message(
        message=Message(
            user_id=user_id,
            chat_id=chat_id,
            role="assistant",
            content=answer_with_timing,
            created_at=datetime.now(),
        )
    )

    chats = get_user_chats(user_id=user_id)
    current_chat = next(
        (chat for chat in chats if chat.chat_id == chat_id),
        None,
    )

    if current_chat is not None:
        current_chat.updated_at = datetime.now()
        update_chat(chat=current_chat)

    # 8. ثبت زمان مراحل
    print(
        "\n===== CONVERSATION TIMINGS =====\n"
        f"Route: {route}\n"
        f"RAG enabled for this request: {rag_used}\n"
        f"Fact extraction/save: "
        f"{timings['fact_extraction_and_save']:.2f}s\n"
        f"Memory retrieval: {timings['memory_retrieval']:.2f}s\n"
        f"Tool routing/execution: "
        f"{timings['tool_routing_and_execution']:.2f}s\n"
        f"RAG retrieval: {timings['rag_retrieval']:.2f}s\n"
        f"Answer generation: {timings['answer_generation']:.2f}s\n"
        f"Total: {total_seconds:.2f}s\n"
    )

    return ConversationResponse(
        chat_id=chat_id,
        message=answer_with_timing,
    )
