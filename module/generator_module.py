
from typing import Final

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


MODEL_NAME: Final[str] = "qwen3.5:4b"

TEMPLATE: Final[str] = """
تو یک دستیار هوشمند فارسی برای پاسخ‌گویی به سؤالات کاربران هستی.

قوانین:
- پاسخ نهایی باید فارسی باشد.
- پاسخ را مستقیم و واضح ارائه کن.
- اطلاعاتی را که در ورودی وجود ندارد، نساز.
- اگر اطلاعات کافی نیست، این موضوع را صریح بیان کن.

متن ورودی:
{question}
"""


def generate_answer(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Generator input must be a non-empty string.")

    prompt = TEMPLATE.format(question=query)

    model = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
        reasoning=False,
        num_predict=512,
    )

    response = model.invoke(
        [HumanMessage(content=prompt)]
    )

    answer = response.content

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(
            f"Generator model '{MODEL_NAME}' returned an empty answer."
        )

    return answer.strip()
