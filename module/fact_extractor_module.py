
import json
from datetime import datetime
from typing import Final, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from module.user_fact_model import UserFact


MODEL_NAME: Final[str] = "gemma3:4b"

INVALID_VALUES: Final[set[str]] = {
    "none",
    "null",
    "unknown",
    "نامشخص",
    "ندارم",
    "نامعلوم",
}


TEMPLATE: Final[str] = """
تو وظیفه داری اطلاعات دائمی و قابل استفاده درباره کاربر را از پیام او استخراج کنی.

فقط اطلاعاتی را استخراج کن که کاربر درباره خودش گفته است.

مثال:

ورودی:
اسم من علی است و برنامه‌نویس هستم.

خروجی:
[
    {{
        "key": "name",
        "value": "علی"
    }},
    {{
        "key": "job",
        "value": "برنامه‌نویس"
    }}
]

مثال دوم:

ورودی:
اسم من چیست؟

خروجی:
[]

مثال سوم:

ورودی:
من علی‌ام

خروجی:
[
    {{
        "key": "name",
        "value": "علی"
    }}
]

قوانین:
- فقط JSON معتبر برگردان.
- خروجی باید یک Array باشد.
- هر مورد فقط شامل key و value باشد.
- اگر هیچ اطلاعاتی درباره خود کاربر وجود ندارد، [] برگردان.
- سؤال درباره اطلاعات قبلی کاربر، Fact جدید نیست.
- اطلاعاتی که درباره افراد دیگر است استخراج نکن.
- برای اطلاعاتی که کاربر نگفته است، مقدار نساز.
- از مقدارهایی مثل None، null و Unknown استفاده نکن.
- اگر شغل مشخص نشده، اصلاً Fact مربوط به شغل تولید نکن.
- نام و شغل را به زبان خود کاربر و بدون توضیح اضافه ثبت کن.
- توضیح اضافه و Markdown ننویس.

پیام کاربر:
{message}

خروجی:
""".strip()


def _clean_json_response(response: str) -> str:
    """پاک‌سازی قالب پاسخ مدل پیش از JSON parsing."""

    response = response.strip()

    if response.startswith("```"):
        lines = response.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        response = "\n".join(lines).strip()

    return response


def _is_valid_fact_value(value: str) -> bool:
    """بررسی معتبر بودن مقدار استخراج‌شده."""

    normalized_value = value.strip().casefold()

    if not normalized_value:
        return False

    if normalized_value in INVALID_VALUES:
        return False

    if normalized_value.startswith("none "):
        return False

    return True


def extract_facts(
    user_id: str,
    message: str,
) -> list[UserFact]:
    """استخراج Factهای قابل ذخیره از پیام کاربر."""

    if not user_id or not user_id.strip():
        return []

    if not message or not message.strip():
        return []

    prompt_template = ChatPromptTemplate.from_template(
        template=TEMPLATE,
    )

    prompt = prompt_template.format(
        message=message.strip(),
    )

    model = OllamaLLM(
        model=MODEL_NAME,
        temperature=0,
    )

    response: str = model.invoke(input=prompt)

    if not isinstance(response, str) or not response.strip():
        return []

    response = _clean_json_response(response)

    try:
        data: Any = json.loads(response)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    now = datetime.now()
    facts: list[UserFact] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        key = item.get("key")
        value = item.get("value")

        if not isinstance(key, str):
            continue

        if not isinstance(value, str):
            continue

        key = key.strip().lower()
        value = value.strip()

        if not key or not _is_valid_fact_value(value):
            continue

        facts.append(
            UserFact(
                user_id=user_id,
                key=key,
                value=value,
                created_at=now,
                updated_at=now,
            )
        )

    return facts
