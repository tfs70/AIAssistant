from dataclasses import dataclass
from datetime import datetime


@dataclass
class Chat:
    """مدل اطلاعات یک Chat"""

    chat_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime