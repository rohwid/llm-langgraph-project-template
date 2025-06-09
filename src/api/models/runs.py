from fastapi import UploadFile
from pydantic import BaseModel, Field
from typing import Optional, List

class SetMessageRequest(BaseModel):
    user_id: str = Field(..., description="The ID of the user sending the message")
    question: str = Field(..., description="The message or question content")

class SentMessageRequest(SetMessageRequest):
    callback_url: str = Field(..., description="The URL to send the callback to")

class MessageModel(BaseModel):
    answer: Optional[str] = Field(None, description="The answer content from the Agent")
    thread_id: str = Field(..., description="The ID of the thread this message belongs to")