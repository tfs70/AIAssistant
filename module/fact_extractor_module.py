import json
from datetime import datetime
from typing import Final

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from module.user_fact_model import UserFact


MODEL_NAME: Final[str] = "gemma3:1b".replace(" ", "").lower()

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

قوانین:
- فقط JSON معتبر برگردان.
- خروجی باید یک Array باشد.
- هر مورد فقط شامل key و value باشد.
- اگر هیچ اطلاعاتی درباره خود کاربر وجود ندارد، [] برگردان.
- سؤال درباره اطلاعات قبلی کاربر، Fact جدید نیست.
- اطلاعاتی که درباره افراد دیگر است استخراج نکن.
- حدس نزن.
- توضیح اضافه ننویس.
- Markdown استفاده نکن.

پیام کاربر:
{message}

خروجی:
""".strip()


def extract_facts(
    user_id: str,
    message: str,
) -> list[UserFact]:
    """استخراج Factهای قابل ذخیره از پیام کاربر"""

    prompt_template = ChatPromptTemplate.from_template(
        template=TEMPLATE,
    )

    prompt: str = prompt_template.format(
        message=message,
    )

    model = OllamaLLM(
        model=MODEL_NAME,
    )

    response: str = model.invoke(
        input=prompt,
    )

    response = response.strip()

    if response.startswith("```"):
        response = response.replace("```json", "")
        response = response.replace("```", "")
        response = response.strip()

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    now: datetime = datetime.now()

    facts: list[UserFact] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        if "key" not in item or "value" not in item:
            continue

        if not isinstance(item["key"], str):
            continue

        if not isinstance(item["value"], str):
            continue

        if not item["key"] or not item["value"]:
            continue

        facts.append(
            UserFact(
                user_id=user_id,
                key=item["key"],
                value=item["value"],
                created_at=now,
                updated_at=now,
            )
        )

    return facts