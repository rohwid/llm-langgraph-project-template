from src.entity.config_entity import ChatbotConfig

from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List

from trustcall import create_extractor

class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    interests: list[str] = Field(description="Interests that the user has", default_factory=list)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list
    )

class ProfileUpdater:
    """
    This class is responsible for updating the user's profile based on the given configuration.
    It initializes the ProfileUpdater with the given configuration.
    """
    def __init__(self, config: ChatbotConfig):
        """
        Initializes the ProfileUpdater with the given configuration.

        Args:
            config (ChatbotConfig): The configuration for the ProfileUpdater.
        """
        self.config = config
        
    def update_profile(self, updated_messages: List[Any], 
                       existing_memories: List[Any]) -> Dict[str, str]:
        """
        Updates the user's profile based on the given messages and existing memories.

        This method initializes a profile extractor using the ChatOpenAI model and Profile tool,
        then processes the updated messages and existing memories to update the user's profile information.
        The extractor analyzes the conversation history and existing profile data to identify
        and extract relevant user information.

        Args:
            updated_messages (List[Any]): List of message objects containing the conversation history
            existing_memories (List[Any]): List of tuples containing existing profile information
                in the format (key, tool_name, value)

        Returns:
            Dict[str, str]: Response containing the extracted and updated profile information
        """
        model = ChatOpenAI(model=self.config.model, temperature=self.config.temperature)
        profile_extractor = create_extractor(model,
                                             tools=[Profile],
                                             tool_choice="Profile")
        
        try:
            response = profile_extractor.invoke({
                "messages": updated_messages, 
                "existing": existing_memories
            })
        except Exception as e:
            logger.error(f"Error invoking profile extractor: {e}")
            raise
        
        return response