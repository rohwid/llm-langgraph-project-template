from langgraph.graph import MessagesState
from pydantic import Field

class QnAChatbotState(MessagesState):
    """Context based on the user's uploaded file and retrieved documents from RAG"""
    retrieved_docs: str = Field(..., description="Retrieved documents from CEdX menu database for RAG.")