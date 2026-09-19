from dataclasses import dataclass
from datetime import datetime


@dataclass
class UserFact:
    """مدل اطلاعات ذخیره شده درباره یک کاربر"""

    user_id: str
    key: str
    value: str
    created_at: datetime
    updated_at: datetime