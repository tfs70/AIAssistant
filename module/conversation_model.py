from dataclasses import dataclass


@dataclass
class ConversationRequest:
    """مدل اطلاعات یک Request"""

    chat_id: str
    user_id: str
    message: str
    
    
    
@dataclass
class ConversationResponse:
    """مدل اطلاعات یک Response"""

    chat_id: str
    message: str