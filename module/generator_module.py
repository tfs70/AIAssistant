from typing import Final

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


MODEL_NAME: Final[str] = "qwen3.5:4b"

TEMPLATE: Final[str] = """
تو یک دستیار هوشمند فارسی‌زبان هستی که به سیستم ERP متصل است.

ورودی ممکن است شامل سؤال کاربر، سابقه مکالمه و نتیجه واقعی ابزارهای ERP باشد.

قوانین قطعی:

1. پاسخ نهایی را حتماً تولید کن؛ سؤال کاربر را به‌عنوان پاسخ تکرار نکن.
2. اگر بخش TRUSTED ERP TOOL RESULT در ورودی وجود دارد،
   از اطلاعات آن برای پاسخ به سؤال کاربر استفاده کن.
3. نتیجه ابزار ERP منبع معتبر اطلاعات عملیاتی است.
4. نام‌ها، شناسه‌ها و مقادیر عددی را دقیقاً مطابق نتیجه ابزار بنویس.
5. هیچ نام، عدد، آمار یا اطلاعاتی را که در نتیجه ابزار وجود ندارد، نساز.
6. اگر نتیجه ابزار برای پاسخ کافی نیست، صریحاً بگو اطلاعات کافی نیست.
7. پاسخ را به زبان فارسی و متناسب با سؤال کاربر بنویس.
8. پاسخ را مستقیم و خوانا ارائه کن؛ درباره مراحل داخلی پردازش توضیح نده.

روش پاسخ‌گویی:
- اگر کاربر فهرستی خواسته است، موارد را به‌صورت فهرست شماره‌دار ارائه کن.
- اگر تعداد مشخصی خواسته است، همان تعداد را از داده‌های موجود ارائه کن.
- فقط اطلاعات مرتبط با سؤال را نمایش بده.
- اگر نتیجه ابزار موجود است، به‌جای گفتن «اطلاعات کافی ندارم»، ابتدا همان نتیجه را بررسی کن.

ورودی:
{question}

پاسخ نهایی فارسی:
""".strip()


def generate_answer(query: str) -> str:
    """تولید پاسخ فارسی با استفاده از مدل محلی Ollama."""

    prompt_template = ChatPromptTemplate.from_template(
        template=TEMPLATE
    )

    prompt = prompt_template.format(question=query)

    model = OllamaLLM(model=MODEL_NAME)

    response = model.invoke(input=prompt)

    if not isinstance(response, str) or not response.strip():
        raise ValueError("Generator returned an empty answer.")

    return response.strip()
