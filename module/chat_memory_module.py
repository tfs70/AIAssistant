import json
import os
import uuid

from datetime import datetime
from typing import Final

from module.chat_model import Chat


DEFAULT_USER_ID: Final[str] = "default"
MEMORY_FILE_PATH: Final[str] = "data/chat_memory.json"


def create_chat(
    user_id: str,
    title: str = "New Chat",
) -> Chat:
    """ایجاد یک Chat جدید برای کاربر"""

    chat_id: str = str(uuid.uuid4())
    now: datetime = datetime.now()

    chat: Chat = Chat(
        chat_id=chat_id,
        user_id=user_id,
        title=title,
        created_at=now,
        updated_at=now,
    )

    return chat


def save_chat(
    chat: Chat,
) -> None:
    """ذخیره یک Chat در فایل حافظه"""

    os.makedirs(
        name=os.path.dirname(MEMORY_FILE_PATH),
        exist_ok=True,
    )

    memory: dict[str, list[dict[str, str]]] = {
        "chats": [],
        "messages": [],
    }

    if os.path.exists(path=MEMORY_FILE_PATH):
        with open(
            file=MEMORY_FILE_PATH,
            mode="r",
            encoding="utf-8",
        ) as file:
            memory = json.load(file)

    chat_data: dict[str, str] = {
        "chat_id": chat.chat_id,
        "user_id": chat.user_id,
        "title": chat.title,
        "created_at": chat.created_at.isoformat(),
        "updated_at": chat.updated_at.isoformat(),
    }

    memory["chats"].append(chat_data)

    with open(
        file=MEMORY_FILE_PATH,
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            memory,
            file,
            ensure_ascii=False,
            indent=4,
        )

def save_message(
    role: str,
    content: str,
    user_id: str,
    chat_id: str,
) -> None:
    """ذخیره یک پیام در فایل حافظه"""

    os.makedirs(
        name=os.path.dirname(MEMORY_FILE_PATH),
        exist_ok=True,
    )

    memory: dict[str, list[dict[str, str]]] = {
        "chats": [],
        "messages": [],
    }

    if os.path.exists(path=MEMORY_FILE_PATH):
        with open(
            file=MEMORY_FILE_PATH,
            mode="r",
            encoding="utf-8",
        ) as file:
            memory = json.load(file)

    message: dict[str, str] = {
        "user_id": user_id,
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "created_at" : datetime.now().isoformat()
    }

    memory["messages"].append(message)

    with open(
        file=MEMORY_FILE_PATH,
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            memory,
            file,
            ensure_ascii=False,
            indent=4,
        )

def get_recent_messages(
    user_id: str,
    chat_id: str,
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
        memory: dict[str, list[dict[str, str]]] = json.load(file)

    messages: list[dict[str, str]] = memory["messages"]

    filtered_messages: list[dict[str, str]] = [
        message
        for message in messages
        if message["user_id"] == user_id
        and message["chat_id"] == chat_id
    ]

    return filtered_messages[-limit:]