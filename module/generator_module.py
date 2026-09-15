from typing import Final

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


MODEL_NAME: Final[str] = "gemma3:1b".replace(" ", "").lower()

TEMPLATE: Final[str] = """
تو یک دستیار هوشمند هستی.

قوانین:
- همیشه به زبان فارسی پاسخ بده.
- فقط بر اساس اطلاعات موجود در مکالمه پاسخ بده.
- اطلاعاتی که کاربر درباره خودش گفته را به خاطر بسپار.
- هرگز اطلاعاتی درباره کاربر که در مکالمه وجود ندارد نساز.
- اگر پاسخ در مکالمه وجود دارد، همان اطلاعات را استفاده کن.
- اگر پاسخ در مکالمه وجود ندارد، بگو «اطلاعات کافی ندارم».

مکالمه:
{question}

پاسخ:
""".strip()


def generate_answer(
    query: str,
) -> str:
    """تولید پاسخ با استفاده از مدل Gemma"""

    prompt_template = ChatPromptTemplate.from_template(
        template=TEMPLATE,
    )

    prompt: str = prompt_template.format(
        question=query,
    )

    model = OllamaLLM(
        model=MODEL_NAME,
    )

    response: str = model.invoke(
        input=prompt,
    )

    return response

