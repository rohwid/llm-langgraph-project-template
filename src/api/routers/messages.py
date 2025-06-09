from fastapi import APIRouter, Form
from langgraph_sdk import get_client

from src.api.config import api_config
from src.api.libs.langgraph_messages import LangGraphMessages
from src.api.models.threads import GetUserIDRequest
from src.api.models.messages import MessagesModel

router = APIRouter()

client = get_client(url=api_config.langgraph_server_url)

@router.post("/get_messages", response_model=MessagesModel)
async def get_messages(request: GetUserIDRequest = Form()):
    """
    Retrieves messages for a given thread ID.

    This endpoint takes a 'GetThreadIDRequest' object as a form parameter, which contains the thread ID.
    It then uses the 'LangGraphMessages' class to fetch the messages associated with the specified thread ID.
    The fetched messages are returned as a 'MessagesModel' object.

    Args:
        request (GetThreadIDRequest): The request object containing the thread ID.

    Returns:
        MessagesModel: A model containing the list of messages for the specified thread ID.
    """
    langgraph_message = LangGraphMessages(
        client=client
    )
    
    messages = await langgraph_message.get_messages(
        user_id=request.user_id
    )
    
    return MessagesModel(messages=messages)