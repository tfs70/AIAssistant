
import json
import os
from typing import Any, Final

from langchain_ollama import OllamaLLM


MODEL_NAME: Final[str] = os.getenv(
    "DECISION_MODEL",
    "qwen3.5:4b",
).strip()

TOOLS: Final[list[dict[str, Any]]] = [
    {
        "name": "get_top_purchased_products",
        "description": (
            "کالاهایی که واقعاً خریداری شده‌اند، "
            "رتبه‌بندی بر اساس مقدار خرید. "
            "با سفارش خرید اشتباه نشود."
        ),
        "arguments": {
            "topCount": "integer اختیاری",
            "brandNames": "آرایه‌ای از نام برندها، اختیاری",
            "vendorNames": "آرایه‌ای از نام تأمین‌کنندگان، اختیاری",
        },
    },
    {
        "name": "get_products_purchase_orders",
        "description": (
            "کالاهای موجود در سفارش‌های خرید، "
            "رتبه‌بندی بر اساس مقدار سفارش. "
            "با خرید قطعی اشتباه نشود."
        ),
        "arguments": {
            "topCount": "integer اختیاری",
            "brandNames": "آرایه‌ای از نام برندها، اختیاری",
            "vendorNames": "آرایه‌ای از نام تأمین‌کنندگان، اختیاری",
        },
    },
    {
        "name": "get_vendor_purchase_share",
        "description": (
            "درصد سهم یک تأمین‌کننده از کل خریدها. "
            "vendorNames الزامی است. "
            "نام تأمین‌کننده باید دقیقاً حفظ شود."
        ),
        "arguments": {
            "vendorNames": "آرایه‌ای از نام تأمین‌کنندگان، الزامی",
        },
    },
    {
        "name": "get_contract_purchase_order_price_mismatch",
        "description": (
            "کالاهایی که قیمت قرارداد خرید با مبلغ "
            "درج‌شده در سفارش خرید مغایرت دارد."
        ),
        "arguments": {
            "topCount": "integer اختیاری",
            "brandNames": "آرایه‌ای از نام برندها، اختیاری",
        },
    },
    {
        "name": "get_upcoming_purchase_orders",
        "description": (
            "سفارش‌های خرید تأییدشده که تاریخ حمل آن‌ها "
            "در تعداد روزهای آینده قرار دارد."
        ),
        "arguments": {
            "topCount": "integer اختیاری",
            "daysAhead": "integer اختیاری",
        },
    },
    {
        "name": "get_vendors_with_highest_delivery_delay",
        "description": (
            "تأمین‌کنندگان با بیشترین مجموع روزهای تأخیر "
            "در رسیدهای خرید."
        ),
        "arguments": {
            "topCount": "integer اختیاری",
        },
    },
]

TOOL_NAMES: Final[set[str]] = {
    tool["name"] for tool in TOOLS
}

SYSTEM_PROMPT: Final[str] = """
تو یک ماژول تصمیم‌گیری برای دستیار ERP هستی.
وظیفه تو فقط انتخاب ابزار مناسب و استخراج آرگومان‌هاست؛
خودت هیچ ابزاری را اجرا نمی‌کنی و پاسخ گزارش ERP را نمی‌سازی.

قوانین:
- سؤال را به زبان فارسی تحلیل کن.
- فقط از ابزارهای ارائه‌شده استفاده کن.
- خرید واقعی با سفارش خرید متفاوت است.
- اگر کاربر از خرید انجام‌شده صحبت کرد، ابزار get_top_purchased_products
  را انتخاب کن.
- اگر کاربر درباره سفارش‌های خرید صحبت کرد، ابزار
  get_products_purchase_orders را انتخاب کن.
- برای سهم درصدی تأمین‌کننده از get_vendor_purchase_share استفاده کن.
- برای مغایرت قیمت قرارداد و سفارش از
  get_contract_purchase_order_price_mismatch استفاده کن.
- برای سفارش‌های حمل در روزهای آینده از
  get_upcoming_purchase_orders استفاده کن.
- برای بیشترین تأخیر تأمین‌کنندگان از
  get_vendors_with_highest_delivery_delay استفاده کن.
- اگر سؤال به هیچ‌یک از ابزارها مربوط نیست، requires_tool=false باشد
  و tool_name برابر null قرار بگیرد.
- نام فارسی برندها و تأمین‌کنندگان را دقیقاً همان‌طور که کاربر نوشته
  حفظ کن؛ ترجمه، اصلاح یا آوانویسی نکن.
- آرگومان‌هایی را که کاربر مشخص نکرده، حدس نزن.
- پارامترهای اختیاریِ مشخص‌نشده را از arguments حذف کن.
- خروجی باید فقط یک JSON معتبر باشد؛ بدون Markdown و توضیح اضافه.

قالب خروجی:
{
  "requires_tool": true,
  "tool_name": "نام دقیق ابزار یا null",
  "arguments": {},
  "confidence": 0.0
}
""".strip()

def _normalize_persian(text: str) -> str:
    """Normalize common Persian/Arabic character variants."""
    return (
        text.casefold()
        .replace("ي", "ی")
        .replace("ك", "ک")
        .replace("\u200c", " ")
        .strip()
    )


def _rule_based_decision(
    query: str,
    model_name: str,
) -> dict[str, Any] | None:
    """Handle unambiguous ERP questions without an LLM call."""

    normalized = _normalize_persian(query)

    delay_terms = (
        "تأخیر",
        "تاخیر",
        "دیرکرد",
        "دیر تحویل",
        "دیرتر تحویل",
    )

    vendor_terms = (
        "تأمین",
        "تامین",
        "فروشنده",
        "supplier",
        "vendor",
    )

    if (
        any(term in normalized for term in delay_terms)
        and any(term in normalized for term in vendor_terms)
    ):
        return {
            "requires_tool": True,
            "tool_name": "get_vendors_with_highest_delivery_delay",
            "arguments": {},
            "confidence": 1.0,
            "model_name": model_name,
            "error": None,
        }

    return None

def decide_tool(
    query: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Select an ERP tool using deterministic rules first, then an LLM."""

    selected_model = model_name or MODEL_NAME

    # Step 1: Handle unambiguous questions deterministically.
    rule_decision = _rule_based_decision(
        query=query,
        model_name=selected_model,
    )

    if rule_decision is not None:
        print(
            "[DECISION] Rule-based selection: "
            f"{rule_decision['tool_name']}"
        )
        return rule_decision

    # Step 2: Use the existing LLM decision logic for other questions.
    prompt = f"""
{SYSTEM_PROMPT}

ابزارهای مجاز:
{json.dumps(TOOLS, ensure_ascii=False, indent=2)}

سؤال کاربر:
{query}

تصمیم JSON:
""".strip()

    # ادامه کد فعلی از اینجا