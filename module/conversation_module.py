
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
from module.rag_module import (
    create_vector_store,
    retrieve_context,
    sync_rag_documents,
)


RouteExecutor = Callable[[str], dict[str, Any]]

VALID_ROUTES = {"chat", "tool", "rag", "tool_and_rag"}

# آستانه اولیه و قابل تنظیم برای تشخیص ارتباط سؤال با اسناد.
# این مقدار باید با آزمون پرسش‌های واقعی تنظیم شود.
RAG_FALLBACK_THRESHOLD = 0.24


def _normalize_persian(text: str) -> str:
    """یکسان‌سازی حروف رایج عربی و فارسی."""
    return (
        text.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("\u200c", "‌")
        .strip()
    )


def _may_contain_personal_fact(message: str) -> bool:
    """تشخیص پیام‌های حاوی اطلاعات شخصی جدید، نه سؤال درباره حافظه."""

    normalized = _normalize_persian(message)

    question_patterns = (
        "اسم من چیه",
        "نام من چیه",
        "اسمم چیه",
        "نامم چیه",
        "اسم من چیست",
        "نام من چیست",
        "شغل من چیه",
        "شغلم چیه",
        "کار من چیه",
        "کارم چیه",
        "شغل من چیست",
        "شغلم چیست",
        "من کی هستم",
        "من چه کسی هستم",
        "چه شغلی دارم",
        "چی درباره من می‌دونی",
        "چه چیزی از من می‌دونی",
        "یادت هست اسمم چیه",
        "یادت هست شغلم چیه",
    )

    if any(pattern in normalized for pattern in question_patterns):
        return False

    if "؟" in normalized or "?" in normalized:
        question_markers = (
            "اسم من",
            "نام من",
            "شغل من",
            "شغلم",
            "اسمم",
            "نامم",
            "کارم",
            "کار من",
        )

        if any(marker in normalized for marker in question_markers):
            return False

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

    if any(marker in normalized for marker in personal_markers):
        return True

    if re.search(r"من\s+\S+\s+(?:ام|هستم)", normalized):
        return True

    if re.search(r"من\s+\S+‌ام", normalized):
        return True

    return False


def _format_tool_results(
    tool_results: list[dict[str, Any]],
) -> str:
    """تبدیل نتایج ساختاریافته ابزارها به متن."""

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
    """ساخت زمینه داده‌های معتبر متناسب با مسیر تصمیم‌گیری."""

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


def _should_fallback_to_rag(
    user_text: str,
) -> tuple[bool, float]:
    """
    بررسی عمومی ارتباط سؤال با اسناد، وقتی مدل مسیر chat را انتخاب کرده است.

    خروجی:
        (is_relevant, best_score)

    در صورت بروز خطا، بررسی ناموفق تلقی می‌شود و مکالمه ادامه می‌یابد.
    """

    try:
        vector_store = create_vector_store()

        matches = vector_store.similarity_search_with_relevance_scores(
            user_text,
            k=3,
        )

        if not matches:
            print("[RAG FALLBACK] No document matches found.")
            return False, -1.0

        scores = [
            float(score)
            for _, score in matches
            if score is not None
        ]

        if not scores:
            print("[RAG FALLBACK] No usable relevance scores.")
            return False, -1.0

        best_score = max(scores)

        print(
            "[RAG FALLBACK] "
            f"best_score={best_score:.4f}, "
            f"threshold={RAG_FALLBACK_THRESHOLD:.4f}"
        )

        return best_score >= RAG_FALLBACK_THRESHOLD, best_score

    except Exception as exc:
        print(f"[RAG FALLBACK] Relevance check failed: {exc!r}")
        return False, -1.0


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
        try:
            extracted_facts = extract_facts(
                user_id=user_id,
                message=user_text,
            )

            print(
                f"[MEMORY] user_id={user_id!r}, "
                f"extracted_facts={extracted_facts!r}"
            )

            for fact in extracted_facts:
                # جلوگیری از ذخیره Fact متعلق به کاربر دیگر
                if fact.user_id != user_id:
                    print(
                        "[MEMORY] Skipping fact with mismatched user_id: "
                        f"{fact.user_id!r}"
                    )
                    continue

                save_fact(fact=fact)

        except Exception as exc:
            # خطای استخراج نباید کل مکالمه را متوقف کند.
            print(
                f"[MEMORY] Fact extraction/save failed for "
                f"user_id={user_id!r}: {exc!r}"
            )

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

    print(
        f"[MEMORY] user_id={user_id!r}, "
        f"loaded_facts={facts!r}"
    )

    timings["memory_retrieval"] = (
        time.perf_counter() - stage_started
    )

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

    if route not in VALID_ROUTES:
        raise ValueError(
            f"Invalid route returned by executor: {route!r}"
        )

    tool_results = routing_result.get("tool_results", [])

    if not isinstance(tool_results, list):
        raise TypeError("tool_results must be a list.")

    timings["tool_routing_and_execution"] = (
        time.perf_counter() - stage_started
    )

    # 4. بررسی عمومی RAG در صورت انتخاب chat توسط مدل.
    # مسیرهای MCP تغییر داده نمی‌شوند.
    rag_fallback_used = False

    if use_rag and route == "chat":
        is_relevant, best_score = _should_fallback_to_rag(user_text)

        if is_relevant:
            route = "rag"
            routing_result["route"] = "rag"
            routing_result["rag_query"] = user_text
            rag_fallback_used = True

            print(
                "[RAG FALLBACK] Relevant document detected. "
                "Switching route from chat to rag. "
                f"score={best_score:.4f}"
            )
        else:
            print(
                "[RAG FALLBACK] Keeping chat route. "
                f"best_score={best_score:.4f}"
            )

    # 5. بازیابی محتوای RAG
    stage_started = time.perf_counter()

    rag_context = ""
    rag_used = use_rag and route in {"rag", "tool_and_rag"}

    if rag_used:
        try:
            sync_result = sync_rag_documents()

            print(
                "\n===== RAG SYNC RESULT =====\n"
                f"Added: {sync_result.get('added', 0)}\n"
                f"Updated: {sync_result.get('updated', 0)}\n"
                f"Skipped: {sync_result.get('skipped', 0)}\n"
                f"Deleted: {sync_result.get('deleted', 0)}\n"
            )

            rag_query = routing_result.get("rag_query") or user_text
            rag_context = retrieve_context(query=rag_query) or ""

        except Exception as exc:
            # خطای RAG نباید باعث از دست رفتن کل مکالمه شود.
            print(f"[RAG] Retrieval/sync failed: {exc!r}")
            rag_context = ""

    timings["rag_retrieval"] = time.perf_counter() - stage_started

    # 6. ساخت زمینه مکالمه
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

    # 7. تولید پاسخ
    stage_started = time.perf_counter()

    answer = generate_answer(final_context)

    timings["answer_generation"] = (
        time.perf_counter() - stage_started
    )

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Generator returned an empty answer.")

    answer = answer.strip()
    total_seconds = time.perf_counter() - total_started

    answer_with_timing = (
        f"{answer}\n\n"
        f"---\n"
        f"⏱️ زمان پاسخ: {total_seconds:.2f} ثانیه"
    )

    # 8. ذخیره پیام‌ها
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

    # 9. ثبت زمان مراحل
    print(
        "\n===== CONVERSATION TIMINGS =====\n"
        f"Route: {route}\n"
        f"RAG enabled for this request: {rag_used}\n"
        f"RAG fallback used: {rag_fallback_used}\n"
        f"Fact extraction/save: "
        f"{timings['fact_extraction_and_save']:.2f}s\n"
        f"Memory retrieval: "
        f"{timings['memory_retrieval']:.2f}s\n"
        f"Tool routing/execution: "
        f"{timings['tool_routing_and_execution']:.2f}s\n"
        f"RAG retrieval: "
        f"{timings['rag_retrieval']:.2f}s\n"
        f"Answer generation: "
        f"{timings['answer_generation']:.2f}s\n"
        f"Total: {total_seconds:.2f}s\n"
    )

    return ConversationResponse(
        chat_id=chat_id,
        message=answer_with_timing,
    )
