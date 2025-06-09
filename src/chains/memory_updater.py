from src.entity.config_entity import ChatbotConfig

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from typing import Literal, TypedDict

class UpdateMemory(TypedDict):
    """Specifies the type of memory to update, currently only supporting 'INSTRUCTION'."""
    update_type: Literal["INSTRUCTION"]

class MemoryUpdater:
    """
    This class is responsible for updating the memory of the system based on the given system message and messages.
    It utilizes the ChatOpenAI model to process the messages and update the memory accordingly.
    """
    def __init__(self, config: ChatbotConfig):
        """
        Initializes the MemoryUpdater with a ChatbotConfig instance.

        Args:
            config (ChatbotConfig): The configuration for the chatbot.
        """
        self.config = config
        
    def update_memory(self, system_msg: str, messages: object) -> object:
        """
        Updates the memory of the system based on the given system message and messages.
        It utilizes the ChatOpenAI model to process the messages and update the memory accordingly.

        Args:
            system_msg (str): The system message to be processed.
            messages (object): The list of messages to be processed.

        Returns:
            object: The response from the ChatOpenAI model after processing the messages.
        """
        model = ChatOpenAI(
            model=self.config.model, 
            temperature=self.config.temperature
        )
        
        try:
            response = model.bind_tools(
                [UpdateMemory], 
                parallel_tool_calls=False
            ).invoke([SystemMessage(content=system_msg)] + messages)
        except Exception as e:
            logger.error(f"Error while updating the memory: {e}")
            raise
        
        return response