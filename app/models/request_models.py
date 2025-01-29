from typing import Optional
from pydantic import BaseModel

class SendChatRequestUser(BaseModel):
    email: str
    department_code: str
    company_code: str

class SendChatRequest(BaseModel):
    user: SendChatRequestUser
    question: str
    chat_id: Optional[str] = None