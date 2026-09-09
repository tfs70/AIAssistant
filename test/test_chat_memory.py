from module.chat_memory_module import (
    create_chat,
    get_recent_messages,
    save_message,
)


# ==========================================
# User 1
# ==========================================

user_1: str = "user_001"

chat_1: dict[str, str] = create_chat(
    user_id=user_1,
    title="گفتگو اول",
)

chat_2: dict[str, str] = create_chat(
    user_id=user_1,
    title="گفتگو دوم",
)


save_message(
    role="user",
    content="پیام مربوط به گفتگوی اول",
    user_id=user_1,
    chat_id=chat_1["chat_id"],
)

save_message(
    role="assistant",
    content="پاسخ مربوط به گفتگوی اول",
    user_id=user_1,
    chat_id=chat_1["chat_id"],
)

save_message(
    role="user",
    content="پیام مربوط به گفتگوی دوم",
    user_id=user_1,
    chat_id=chat_2["chat_id"],
)


# ==========================================
# User 2
# ==========================================

user_2: str = "user_002"

chat_3: dict[str, str] = create_chat(
    user_id=user_2,
    title="گفتگوی کاربر دوم",
)

save_message(
    role="user",
    content="پیام مربوط به کاربر دوم",
    user_id=user_2,
    chat_id=chat_3["chat_id"],
)


# ==========================================
# Test Chat 1
# ==========================================

messages: list[dict[str, str]] = get_recent_messages(
    user_id=user_1,
    chat_id=chat_1["chat_id"],
    limit=5,
)

print("\nChat 1:")
for message in messages:
    print(message)


# ==========================================
# Test Chat 2
# ==========================================

messages = get_recent_messages(
    user_id=user_1,
    chat_id=chat_2["chat_id"],
    limit=5,
)

print("\nChat 2:")
for message in messages:
    print(message)


# ==========================================
# Test Chat 3
# ==========================================

messages = get_recent_messages(
    user_id=user_2,
    chat_id=chat_3["chat_id"],
    limit=5,
)

print("\nChat 3:")
for message in messages:
    print(message)