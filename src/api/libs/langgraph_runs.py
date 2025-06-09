from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from loguru import logger
from typing import Any, Dict, List

from src.entity.config_entity import ChatbotConfig

import httpx
import json

class LangGraphRuns:
    """Represents a LangGraph run for processing user queries and files."""
    def __init__(self, api_config: ChatbotConfig,
                 client: object,
                 user_id: str,
                 question: str,
                 thread_id: str):
        """
        Initializes a LangGraphRuns instance with the provided configuration and parameters.

        Args:
            api_config (ChatbotConfig): The configuration for the API.
            client (object): The client object for interacting with the LangGraph service.
            user_id (str): The ID of the user initiating the LangGraph run.
            question (str): The question or message content from the user.
            thread_id (str): The ID of the thread or conversation related to the query.
        """
        self.api_config = api_config
        self.client = client
        self.user_id = user_id
        self.question = question
        self.thread_id = thread_id
        
    async def get_answer(self) -> str:
        """
        Retrieves the answer to the user's question from the LangGraph service.
        This method initiates a LangGraph run to process the user's question and any associated file. 
        
        It sent the whole final answer content without chunking.

        Returns:
            str: The content of the final answer message from the LangGraph service.
        """
        config = {
            "configurable": {
                "user_id": self.user_id
            }
        }
        
        try:
            logger.info(f"Getting answer message for user ID \"{self.user_id}\"..")
            messages = [chunk async for chunk in self.client.runs.stream(
                self.thread_id, 
                self.api_config.graph_name, 
                input={
                    "messages": [HumanMessage(content=self.question)]
                },
                config=config,
                stream_mode="values"
            )]
        except Exception as e:
            logger.error(f"Failed to get answer for user ID \"{self.user_id}\": {str(e)}")
            raise HTTPException(status_code=500, 
                                detail=f"Failed to get answer for user ID \"{self.user_id}\": {str(e)}")
        finally:
            logger.info(f"Getting the answer for user ID \"{self.user_id}\".. done")
        
        return messages[-1].data["messages"][-1]["content"]
    
    async def get_stream(self) -> object:
        """
        Initiates a stream of messages related to the user's question and associated file.

        This method starts a LangGraph run to process the user's question and any associated file. 
        It sent final answer in the form of chunked words including the question, file details, and any subsequent messages 
        exchanged within the thread.

        Returns:
            object: An asynchronous generator that yields a stream of messages related to the user's question.
        """
        config = {
            "configurable": {
                "user_id": self.user_id
            }
        }
        
        try:
            logger.info(f"Streaming answer for user ID \"{self.user_id}\"..")
            async def event_generator():
                async for chunk in self.client.runs.stream(
                    self.thread_id, 
                    self.api_config.graph_name, 
                    input={
                        "messages": [HumanMessage(content=self.question)]
                    },
                    config=config, 
                    stream_mode="messages-tuple"
                ):
                    if chunk.event == "messages":
                        for data in chunk.data:
                            if ("content" in data and 
                                "tool_calls" not in data["additional_kwargs"] and 
                                data["type"] == "AIMessageChunk"):
                                json_chunk = json.dumps({
                                    "answer": data["content"]
                                })
                        
                                # DEBUG: Check chunk data fields
                                # logger.info(f"Chunk: {json.dumps(chunk.data)}")
                                
                                yield f"data: {json_chunk}\n"
        except Exception as e:
            logger.error(f"Failed to stream answer for user ID \"{self.user_id}\" in thread ID \"{self.thread_id}\": {str(e)}")
            raise HTTPException(status_code=500, 
                                detail=f"Failed to stream answer for user ID \"{self.user_id}\" in thread ID \"{self.thread_id}\": {str(e)}")
        finally:
            logger.info(f"Streaming answer for user ID \"{self.user_id}\" in thread ID \"{self.thread_id}\".. done")
        
        return event_generator()
    
    async def _sent_request(self, callback_url: str, data: Dict[str, Any]) -> None:
        """
        Sends a POST request to the specified callback URL with the provided data.

        This method uses the httpx library to send a POST request to the specified callback URL. 
        The request body is the provided data, which is expected to be a JSON string. 
        If the request is successful, it returns the response. 
        If the request fails, it logs the error and raises an HTTPException.

        Args:
            callback_url (str): The URL to which the POST request will be sent.
            data (str): The data to be sent in the request body. It should be a JSON string.

        Raises:
            HTTPException: If the POST request fails.
        """
        try:
            # follow redirects handle / behind the target endpoint
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.post(callback_url, json=data)
                response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"Webhook delivery failed: {str(e)}")
            raise
    
    async def set_requests(self, callback_url: str) -> None:
        """
        Streams the answer for a given user ID to a specified callback URL.

        This method sets up a streaming process to deliver chunks of the answer to a user's question to a specified callback URL. 
        It first configures the streaming process with the user's ID and then attempts to stream the answer. 
        For each chunk of the answer received, it sends the chunk to the callback URL. 
        If any error occurs during the streaming process, it logs the error and raises an HTTPException. 
        Finally, it logs the total number of chunks delivered and completes the delivery process.

        Args:
            callback_url (str): The URL to which the answer chunks will be sent.

        Raises:
            HTTPException: If an error occurs during the streaming process.
        """
        config = {
            "configurable": {
                "user_id": self.user_id
            }
        }
        
        try:
            logger.info(f"Streaming answer for user ID \"{self.user_id}\"..")
            async def event_generator():
                async for chunk in self.client.runs.stream(
                    self.thread_id, 
                    self.api_config.graph_name, 
                    input={
                        "messages": [HumanMessage(content=self.question)]
                    },
                    config=config, 
                    stream_mode="messages-tuple"
                ):
                    if chunk.event == "messages":
                        for data in chunk.data:
                            if ("content" in data and 
                                "tool_calls" not in data["additional_kwargs"] and 
                                data["type"] == "AIMessageChunk"):
                                json_chunk = {
                                    "answer": data["content"]
                                }
                        
                                # DEBUG: Check chunk data fields
                                # logger.info(f"Chunk: {json.dumps(chunk.data)}")
                                
                                yield json_chunk
        except Exception as e:
            logger.error(f"Failed to stream answer for user ID \"{self.user_id}\" in thread ID \"{self.thread_id}\": {str(e)}")
            raise HTTPException(status_code=500, 
                                detail=f"Failed to stream answer for user ID \"{self.user_id}\" in thread ID \"{self.thread_id}\": {str(e)}")
        finally:
            logger.info(f"Streaming answer for user ID \"{self.user_id}\" to \" {callback_url}\".. done")
        
        logger.info(f"Delivering chunks for user ID \"{self.user_id}\" to \" {callback_url}\"..")
        chunks = 0
        
        await self._sent_request(callback_url, {
            "answer": "<answer>"
        })
        
        async for data in event_generator():
            await self._sent_request(callback_url, data)
            chunks += 1
            
        await self._sent_request(callback_url, {
            "answer": "</answer>"
        })
        
        logger.info(f"Total {chunks} chunk for user ID \"{self.user_id}\".")
        logger.info(f"Delivering chunks for user ID \"{self.user_id}\" to \" {callback_url}\".. done")