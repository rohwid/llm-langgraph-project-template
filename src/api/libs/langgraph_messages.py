from fastapi import HTTPException
from loguru import logger
from typing import Dict, List, Optional

from src.api.libs.langgraph_threads import LangGraphThreads

class LangGraphMessages:
    """
    Represents a collection of messages within a thread. 
    
    This class is responsible for managing and retrieving messages within a specific thread. 
    It provides methods to fetch messages, pair them into questions and answers, 
    and associate file URLs and names with each question and answer based on the thread's metadata.
    """
    def __init__(self, client: object):
        """
        Initializes the LangGraphMessages object.

        Args:
            client (object): The client object used for communication.
        """
        self.client = client
    
    async def _init_message(self) -> Dict[str, str]:
        return {
            "question": None,
            "answer": None
        }
        
    async def get_messages(self, user_id: str) -> Optional[List[Dict[str, str]]]:
        """
        Retrieves messages for a given thread ID.

        This method fetches the thread state for the specified thread ID, 
        filters out messages of type 'ai' or 'human' that do not contain 'tool_calls' in their additional keywords, 
        and pairs them into questions and answers. It also associates file URLs and names with 
        each question and answer based on the thread's metadata. 
        
        The method logs the process of fetching thread state and loading messages, 
        and raises an HTTPException if it fails to get the thread state.

        Returns:
            List[dict]: A list of dictionaries, each representing a question-answer pair with their content, 
                        file URLs, and file names.
        """
        langgraph_threads = LangGraphThreads(
            client=self.client
        )
        
        thread_id = await langgraph_threads._search_thread(user_id)
        
        if thread_id:
            try:
                logger.info(f"Getting thread state for thread ID \"{thread_id}\"..")
                thread_state = await self.client.threads.get_state(thread_id)
            except Exception as e:
                logger.exception(f"Failed to get thread state for thread ID \"{thread_id}\": {str(e)}")
                raise HTTPException(status_code=500, 
                                    detail=f"Failed to get thread state thread ID \"{thread_id}\": {str(e)}")
            finally:
                logger.info(f"Getting thread state for thread ID \"{thread_id}\".. done")
            
            logger.info(f"Loading messages for thread ID \"{thread_id}\"..")
            
            messages = []
            set_message = await self._init_message()
            
            for message in thread_state["values"]["messages"]:
                if message["type"] == "human" and "tool_calls" not in message["additional_kwargs"]:
                    set_message["question"] = message["content"]
                    
                if message["type"] == "ai" and "tool_calls" not in message["additional_kwargs"]:
                    set_message["answer"] = message["content"]
                    messages.append(set_message)
                    set_message = await self._init_message()
            
            logger.info(f"Loading messages for thread ID \"{thread_id}\".. done")
        else:
            logger.info(f"No messages for user ID \"{user_id}\".")
        
        return messages