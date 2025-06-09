from fastapi import APIRouter, Form
from langgraph_sdk import get_client

from src.api.config import api_config
from src.api.libs.langgraph_threads import LangGraphThreads
from src.api.models.threads import GetUserIDRequest, ThreadsModel

router = APIRouter()

client = get_client(url=api_config.langgraph_server_url)

@router.post("/get_threads", response_model=ThreadsModel)
async def get_threads(request: GetUserIDRequest = Form()):
    """
    Retrieves a list of threads for a given user ID.

    This endpoint takes a `GetThreadsRequest` object as a form parameter, which contains the user ID, limit, and offset.
    It then uses the `LangGraphThreads` class to fetch the list of threads associated with the specified user ID, 
    applying the limit and offset for pagination. The fetched threads are returned as a `ThreadsModel` object.

    Args:
        request (GetThreadsRequest): The request object containing the user ID, limit, and offset.

    Returns:
        ThreadsModel: A model containing the list of threads for the specified user ID.
    """
    langgraph_threads = LangGraphThreads(
        client=client
    )
    
    threads = await langgraph_threads.get_threads(
        user_id=request.user_id
    )
    
    return ThreadsModel(threads=threads)

@router.post("/delete_thread")
async def delete_thread(request: GetUserIDRequest = Form()):
    """
    Deletes a thread associated with a given user ID.

    This endpoint takes a `GetUserIDRequest` object as a form parameter, which contains the user ID.
    It then uses the `LangGraphThreads` class to delete the thread associated with the specified user ID.
    The ID of the deleted thread is returned in the response.

    Args:
        request (GetUserIDRequest): The request object containing the user ID.

    Returns:
        dict: A dictionary containing the ID of the deleted thread.
    """
    langgraph_threads = LangGraphThreads(
        client=client
    )
    
    thread_id = await langgraph_threads.delete_thread(
        user_id=request.user_id
    )
    
    return {
        "deleted_thread": thread_id
    }

@router.post("/delete_threads")
async def delete_threads(request: GetUserIDRequest = Form()):
    """
    Deletes threads associated with a given user ID.

    This endpoint takes a `GetUserIDRequest` object as a form parameter, which contains the user ID.
    It then uses the `LangGraphThreads` class to delete all threads associated with the specified user ID.
    The IDs of the deleted threads are returned in the response.

    Args:
        request (GetUserIDRequest): The request object containing the user ID.

    Returns:
        dict: A dictionary containing the IDs of the deleted threads.
    """
    langgraph_thread = LangGraphThreads(
        client=client
    )
    
    threads = await langgraph_thread.delete_threads(
        user_id=request.user_id
    )
    
    return {
        "deleted_threads": threads
    }