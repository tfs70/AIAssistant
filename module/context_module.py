from module.message_model import Message
from module.user_fact_model import UserFact


def build_context(
    messages: list[Message],
    facts: list[UserFact],
    rag_context: str,
    current_message: str,
) -> str:
    """ساخت Context نهایی برای Generator"""

    sections: list[str] = []

    # -------------------------------------------------
    # Long Memory
    # -------------------------------------------------

    if facts:
        fact_lines: list[str] = [
            f"{fact.key}: {fact.value}"
            for fact in facts
        ]

        sections.append(
            "===== LONG MEMORY =====\n"
            + "\n".join(fact_lines)
        )

    # -------------------------------------------------
    # RAG
    # -------------------------------------------------

    if rag_context.strip():
        sections.append(
            "===== RAG CONTEXT =====\n"
            + rag_context.strip()
        )

    # -------------------------------------------------
    # Short Memory
    # -------------------------------------------------

    if messages:
        conversation_lines: list[str] = []

        for message in messages:
            conversation_lines.append(
                f"{message.role}: {message.content}"
            )

        sections.append(
            "===== CONVERSATION =====\n"
            + "\n".join(conversation_lines)
        )

    # -------------------------------------------------
    # Current Message
    # -------------------------------------------------

    sections.append(
        "===== CURRENT MESSAGE =====\n"
        + current_message
    )

    return "\n\n".join(sections)