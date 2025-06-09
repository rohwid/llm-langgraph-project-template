from pydantic import BaseModel, Field
from typing import List

class GetUserIDRequest(BaseModel):
    user_id: str = Field(..., description="The ID of the user requesting the threads")

class ThreadsModel(BaseModel):
    threads: List[str] = Field(..., description="List of thread IDs associated with the user")