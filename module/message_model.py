
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:
    user_id: str
    chat_id: str
    role: str
    content: str
    created_at: datetime
