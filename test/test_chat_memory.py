from datetime import datetime

from module.chat_memory_module import (
    create_chat,
    get_recent_messages,
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


save_message(
    message=Message(
        user_id=user_1,
        chat_id=chat_1.chat_id,
        role="user",
        content="پیام مربوط به گفتگوی اول",
        created_at=datetime.now(),
    ),
)

save_message(
    message=Message(
        user_id=user_1,
        chat_id=chat_1.chat_id,
        role="assistant",
        content="پاسخ مربوط به گفتگوی اول",
        created_at=datetime.now(),
    ),
)

save_message(
    message=Message(
        user_id=user_1,
        chat_id=chat_2.chat_id,
        role="user",
        content="پیام مربوط به گفتگوی دوم",
        created_at=datetime.now(),
    ),
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

save_message(
    message=Message(
        user_id=user_2,
        chat_id=chat_3.chat_id,
        role="user",
        content="پیام مربوط به کاربر دوم",
        created_at=datetime.now(),
    ),
)


# ==========================================
# Test Chat 1
# ==========================================

messages: list[Message] = get_recent_messages(
    user_id=user_1,
    chat_id=chat_1.chat_id,
    limit=5,
)

print("\nChat 1:")

for message in messages:
    print(
        f"[{message.created_at}] "
        f"{message.role}: "
        f"{message.content}"
    )


# ==========================================
# Test Chat 2
# ==========================================

messages = get_recent_messages(
    user_id=user_1,
    chat_id=chat_2.chat_id,
    limit=5,
)

print("\nChat 2:")

for message in messages:
    print(
        f"[{message.created_at}] "
        f"{message.role}: "
        f"{message.content}"
    )


# ==========================================
# Test Chat 3
# ==========================================

messages = get_recent_messages(
    user_id=user_2,
    chat_id=chat_3.chat_id,
    limit=5,
)

print("\nChat 3:")

for message in messages:
    print(
        f"[{message.created_at}] "
        f"{message.role}: "
        f"{message.content}"
    )

