from typing import Final

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


MODEL_NAME: Final[str] = "gemma3:1b".replace(" ", "").lower()

TEMPLATE: Final[str] = """
You are a helpful assistant.

Question:
{question}

Answer:
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