import json
import os

from typing import Final


DEFAULT_USER_ID: Final[str] = "default"
MEMORY_FILE_PATH: Final[str] = "data/chat_memory.json"


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


def save_message(
    role: str,
    content: str,
    user_id: str = DEFAULT_USER_ID,
    chat_id: str = "",
) -> None:
    """ذخیره یک پیام در فایل حافظه"""

    os.makedirs(
        name=os.path.dirname(MEMORY_FILE_PATH),
        exist_ok=True,
    )

    messages: list[dict[str, str]] = []

    if os.path.exists(path=MEMORY_FILE_PATH):
        with open(
            file=MEMORY_FILE_PATH,
            mode="r",
            encoding="utf-8",
        ) as file:
            messages = json.load(file)

    message: dict[str, str] = {
        "user_id": user_id,
        "chat_id": chat_id,
        "role": role,
        "content": content,
    }

    messages.append(message)

    with open(
        file=MEMORY_FILE_PATH,
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            messages,
            file,
            ensure_ascii=False,
            indent=4,
        )


def get_recent_messages(
    user_id: str = DEFAULT_USER_ID,
    chat_id: str = "",
    limit: int = 5,
) -> list[dict[str, str]]:
    """دریافت آخرین پیام‌های یک Chat"""

    if not os.path.exists(path=MEMORY_FILE_PATH):
        return []

    with open(
        file=MEMORY_FILE_PATH,
        mode="r",
        encoding="utf-8",
    ) as file:
        messages: list[dict[str, str]] = json.load(file)

    filtered_messages: list[dict[str, str]] = [
        message
        for message in messages
        if message["user_id"] == user_id
        and message["chat_id"] == chat_id
    ]

    return filtered_messages[-limit:]