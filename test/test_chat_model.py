from datetime import datetime

from module.chat_model import Chat


now: datetime = datetime.now()

chat: Chat = Chat(
    chat_id="chat_001",
    user_id="user_001",
    title="گفتگو درباره فروش",
    created_at=now,
    updated_at=now,
)

print(chat)