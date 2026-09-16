from module.message_model import Message


def build_context(
    messages: list[Message],
    current_message: str,
) -> str:
    """ساخت Context برای Generator"""

    context: str = ""

    for message in messages:
        context += f"{message.role}: {message.content}\n"

    context += f"user: {current_message}"

    return context