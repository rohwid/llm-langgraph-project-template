from pydantic import BaseModel, Field
from typing import Dict, List

class MessagesModel(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="List of messages containing questions and answers")