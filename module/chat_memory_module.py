from typing import Final


DEFAULT_USER_ID: Final[str] = "default"


def create_chat(
    user_id: str = DEFAULT_USER_ID,
    title: str = "New Chat",
) -> dict[str, str]:
    """ایجاد یک Chat جدید برای کاربر"""

    import uuid

    chat_id: str = str(uuid.uuid4())

    chat: dict[str, str] = {
        "chat_id": chat_id,
        "user_id": user_id,
        "title": title,
    }

    return chat