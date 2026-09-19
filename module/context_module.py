from module.message_model import Message
from module.user_fact_model import UserFact


def build_context(
    messages: list[Message],
    facts: list[UserFact],
    current_message: str,
) -> str:
    """ساخت Context برای Generator"""

    context: str = ""

    # -------------------------------------------------
    # Long Memory
    # -------------------------------------------------

    if facts:
        context += "user facts:\n"

        for fact in facts:
            context += f"{fact.key}: {fact.value}\n"

        context += "\n"

    # -------------------------------------------------
    # Short Memory
    # -------------------------------------------------

    if messages:
        context += "conversation:\n"

        for message in messages:
            context += f"{message.role}: {message.content}\n"

        context += "\n"

    # -------------------------------------------------
    # Current Message
    # -------------------------------------------------

    context += f"user: {current_message}"

    return context