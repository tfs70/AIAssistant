from module.chat_memory_module import create_chat


chat: dict[str, str] = create_chat(
    user_id="user_001",
    title="گفتگو درباره فروش",
)

print(chat)