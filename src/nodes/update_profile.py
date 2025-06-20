from datetime import datetime
from langgraph.store.base import BaseStore
from langchain_core.messages import merge_message_runs
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from typing import Any, Dict, List

from src.chains.profile_updater import ProfileUpdater
from src.constants import UPDATE_PROFILE_TOOL
from src.config.configuration import Configuration, ConfigurationManager
from src.states import QnAChatbotState
from src.prompts import TRUSTCALL_INSTRUCTION

import uuid

def update_profile(state: QnAChatbotState, config: RunnableConfig, store: BaseStore) -> Dict[str, List[Any]]:
    """
    Updates the user's profile by processing the chat history and incorporating new information.

    This function updates the user's profile by processing the chat history and incorporating new information.
    It retrieves the user's ID from the configuration, defines a namespace for the memories, and retrieves existing memories.
    The function then formats the existing memories for the Trustcall extractor, merges the chat history with the TRUSTCALL instruction,
    and updates the profile using the ProfileUpdater. Finally, it saves the updated memories to the store and returns a message indicating
    the update was successful.

    Args:
        state (MettaPrompterState): The state containing the chat history and other relevant information.
        config (RunnableConfig): The configuration for the profile update process.
        store (BaseStore): The store where the memories are saved.

    Returns:
        Dict[str, List[Any]]: A dictionary containing a message indicating the profile update was successful.
    """
    config_manager = ConfigurationManager()
    chatbot_config = config_manager.get_chatbot_config()
    
    # Get the user ID from the config
    configurable = Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Profile"
    
    existing_memories = None
    if existing_items:
        existing_memories = [(existing_item.key, 
                              tool_name, 
                              existing_item.value) for existing_item in existing_items]

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    
    # Merge the system message with TRUSTCALL instruction with all message from chat history except the latest or last
    updated_messages=list(merge_message_runs(
        messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]
    ))
    
    logger.info(f"Updating profile for user ID \"{user_id}\"..")
    
    profile_updater = ProfileUpdater(config=chatbot_config)
    result = profile_updater.update_profile(updated_messages, existing_memories)

    # Save save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"))
        
    tool_calls = state["messages"][-1].tool_calls
    
    logger.info(f"Updating profile for user ID \"{user_id}\".. done")
    
    # Return tool message with update verification
    return {
        "messages": [{
            "role": "tool", 
            "name": UPDATE_PROFILE_TOOL,
            "content": "Profile updated.", 
            "tool_call_id": tool_calls[0]["id"]
        }]
    }