from datetime import datetime
from time import sleep

from module.chat_memory_module import (
    create_chat,
    get_chat_messages,
    get_recent_messages,
    get_user_chats,
    save_chat,
    save_message,
)
from module.message_model import Message


# ==========================================
# User 1
# ==========================================

user_1: str = "user_001"

chat_1 = create_chat(
    user_id=user_1,
    title="گفتگو اول",
)

save_chat(
    chat=chat_1,
)

chat_2 = create_chat(
    user_id=user_1,
    title="گفتگو دوم",
)

save_chat(
    chat=chat_2,
)


# ==========================================
# User 2
# ==========================================

user_2: str = "user_002"

chat_3 = create_chat(
    user_id=user_2,
    title="گفتگوی کاربر دوم",
)

save_chat(
    chat=chat_3,
)


# ==========================================
# Save Messages
# ==========================================

message_1 = Message(
    user_id=user_1,
    chat_id=chat_1.chat_id,
    role="user",
    content="پیام اول کاربر",
    created_at=datetime.now(),
)

save_message(
    message=message_1,
)

sleep(1)

message_2 = Message(
    user_id=user_1,
    chat_id=chat_1.chat_id,
    role="assistant",
    content="پاسخ اول دستیار",
    created_at=datetime.now(),
)

save_message(
    message=message_2,
)

sleep(1)

message_3 = Message(
    user_id=user_1,
    chat_id=chat_2.chat_id,
    role="user",
    content="پیام مربوط به گفتگوی دوم",
    created_at=datetime.now(),
)

save_message(
    message=message_3,
)

sleep(1)

message_4 = Message(
    user_id=user_2,
    chat_id=chat_3.chat_id,
    role="user",
    content="پیام مربوط به کاربر دوم",
    created_at=datetime.now(),
)

save_message(
    message=message_4,
)


# ==========================================
# Test Recent Messages
# ==========================================

messages: list[Message] = get_recent_messages(
    user_id=user_1,
    chat_id=chat_1.chat_id,
    limit=5,
)

print("\nRecent Messages - Chat 1:")

for message in messages:
    print(
        message.role,
        "|",
        message.created_at,
        "|",
        message.content,
    )


# ==========================================
# Test All Chat Messages
# ==========================================

messages = get_chat_messages(
    user_id=user_1,
    chat_id=chat_1.chat_id,
)

print("\nAll Messages - Chat 1:")

for message in messages:
    print(
        message.role,
        "|",
        message.created_at,
        "|",
        message.content,
    )


# ==========================================
# Test User Chats
# ==========================================

chats = get_user_chats(
    user_id=user_1,
)

print("\nUser 1 Chats:")

for chat in chats:
    print(
        chat.title,
        "|",
        chat.updated_at,
    )


# ==========================================
# Test User Isolation
# ==========================================

chats = get_user_chats(
    user_id=user_2,
)

print("\nUser 2 Chats:")

for chat in chats:
    print(
        chat.title,
        "|",
        chat.updated_at,
    )


# ==========================================
# Test Chat Update
# ==========================================

print("\nChat 1 Updated At:")

for chat in get_user_chats(user_id=user_1):
    if chat.chat_id == chat_1.chat_id:
        print(
            chat.title,
            "|",
            chat.updated_at,
        )
