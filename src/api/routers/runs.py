from fastapi import APIRouter, BackgroundTasks, Form
from fastapi.responses import StreamingResponse
from langgraph_sdk import get_client

from src.api.config import api_config
from src.api.libs.langgraph_runs import LangGraphRuns
from src.api.libs.langgraph_threads import LangGraphThreads
from src.api.models.runs import SetMessageRequest, SentMessageRequest, MessageModel

router = APIRouter()

client = get_client(url=api_config.langgraph_server_url)

@router.post("/sent_message", response_model=MessageModel)
async def sent_message(request: SentMessageRequest = Form(), 
                       background_tasks: BackgroundTasks = None):
    """
    Handles the setting of a message with an optional file upload.
    If a file is uploaded, it is stored in Azure Blob Storage. 
    If no thread_id is provided, a new thread is created with 
    the question as the title.

    Args:
        user_id (str): The ID of the user sending the message
        question (str): The message/question content
        thread_id (str, optional): The ID of an existing thread. Defaults to None.
        callback_url (str): The URL to call back after processing the message.
    
    Returns:
        MessageModel: A model containing the empty answer, 
                      thread_id, thread_title, file_url, file_name, and
                      callback_url.

    Example curl command to call this endpoint:
        curl -X POST "http://localhost:8000/sent_message/" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "user_id=22a0c72f-36e6-4651-b5d7-f03c5b10e26e" \
        -F "question=What is this file?" \
        -F "thread_id=" \
        -F "callback_url=http://example.com/receive_message"
    """    
    langgraph_threads = LangGraphThreads(
        client=client
    )
    
    thread_id = await langgraph_threads.get_thread(
        user_id=request.user_id
    )
    
    runs = LangGraphRuns(
        api_config=api_config,
        client=client,
        user_id=request.user_id,
        question=request.question,
        thread_id=thread_id
    )
    
    background_tasks.add_task(
        runs.set_requests, 
        request.callback_url
    )
    
    return MessageModel(answer="", thread_id=thread_id)

@router.post("/stream_message", response_model=MessageModel)
async def stream_message(request: SetMessageRequest = Form()):
    """
    Handles the setting of a message with an optional file upload.
    If a file is uploaded, it is stored in Azure Blob Storage. 
    If no thread_id is provided, a new thread is created with 
    the question as the title.

    Args:
        user_id (str): The ID of the user sending the message
        question (str): The message/question content
        thread_id (str, optional): The ID of an existing thread. Defaults to None.

    Returns:
        MessageModel: A model containing the answer, thread_id, thread_title, file_url, and file_name.

    Example curl command to call this endpoint:
        curl -X POST "http://localhost:8000/stream_message/" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "user_id=22a0c72f-36e6-4651-b5d7-f03c5b10e26e" \
        -F "question=What is this file?" \
        -F "thread_id=" 
    """
    langgraph_threads = LangGraphThreads(
        client=client
    )
    
    thread_id = await langgraph_threads.get_thread(
        user_id=request.user_id
    )
    
    runs = LangGraphRuns(
        api_config=api_config,
        client=client,
        user_id=request.user_id,
        question=request.question,
        thread_id=thread_id
    )
    
    event_generator = await runs.get_stream()
    
    return StreamingResponse(event_generator, media_type="application/x-ndjson")

@router.post("/set_message", response_model=MessageModel)
async def set_message(request: SetMessageRequest = Form()):
    """
    Handles the setting of a message with an optional file upload.
    If a file is uploaded, it is stored in Azure Blob Storage. 
    If no thread_id is provided, a new thread is created with 
    the question as the title.

    Args:
        user_id (str): The ID of the user sending the message
        question (str): The message/question content
        thread_id (str, optional): The ID of an existing thread. Defaults to None.
        upload_file (UploadFile, optional): File to be uploaded. Defaults to None.

    Returns:
        MessageModel: A model containing the answer, thread_id, thread_title, file_url, and file_name.

    Example curl command to call this endpoint:
        curl -X POST "http://localhost:8000/set_message/" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "user_id=22a0c72f-36e6-4651-b5d7-f03c5b10e26e" \
        -F "question=What is this file?" \
        -F "thread_id="
    """
        
    langgraph_threads = LangGraphThreads(
        client=client
    )
    
    thread_id = await langgraph_threads.get_thread(
        user_id=request.user_id
    )
    
    runs = LangGraphRuns(
        api_config=api_config,
        client=client,
        user_id=request.user_id,
        question=request.question,
        thread_id=thread_id
    )
    
    answer = await runs.get_answer()
        
    return MessageModel(answer=answer, thread_id=thread_id)