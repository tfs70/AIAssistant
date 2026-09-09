import json
import os
import uuid

from datetime import datetime
from typing import Final

from module.chat_model import Chat
from module.message_model import Message


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
    """ذخیره یا بروزرسانی یک Chat در فایل حافظه"""

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

    existing_chat = next(
        (
            item
            for item in memory["chats"]
            if item["chat_id"] == chat.chat_id
            and item["user_id"] == chat.user_id
        ),
        None,
    )

    if existing_chat:
        existing_chat.update(chat_data)
    else:
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
    message: Message,
) -> None:
    """ذخیره یک Message در فایل حافظه"""

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

    message_data: dict[str, str] = {
        "user_id": message.user_id,
        "chat_id": message.chat_id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
    }

    memory["messages"].append(message_data)

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
) -> list[Message]:
    """دریافت آخرین پیام‌های یک Chat"""

    if not os.path.exists(path=MEMORY_FILE_PATH):
        return []

    with open(
        file=MEMORY_FILE_PATH,
        mode="r",
        encoding="utf-8",
    ) as file:
        memory: dict[str, list[dict[str, str]]] = json.load(file)

    messages: list[Message] = [
        Message(
            user_id=message["user_id"],
            chat_id=message["chat_id"],
            role=message["role"],
            content=message["content"],
            created_at=datetime.fromisoformat(message["created_at"]),
        )
        for message in memory["messages"]
        if message["user_id"] == user_id
        and message["chat_id"] == chat_id
    ]

    messages.sort(
        key=lambda message: message.created_at,
    )

    return messages[-limit:]
