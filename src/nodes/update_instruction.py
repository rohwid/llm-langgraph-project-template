from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from loguru import logger
from typing import Dict

from src.chains.instruction_updater import InstructionUpdater
from src.config.configuration import Configuration, ConfigurationManager
from src.constants import UPDATE_INSTRUCTIONS_TOOL
from src.prompts import CREATE_INSTRUCTIONS
from src.states import QnAChatbotState

def update_instructions(state: QnAChatbotState, config: RunnableConfig, store: BaseStore) -> Dict[str, list]:
    """
    Updates the user's instructions based on the chat history.

    This function updates the user's instructions by processing the chat history. It extracts the user's ID from the configuration,
    defines a namespace for the instructions, and formats the existing instructions for the system prompt. It then updates the
    instructions using the InstructionUpdater and saves the updated instructions to the store. Finally, it returns a message
    indicating the update was successful.

    Args:
        state (MettaPrompterState): The state containing the chat history and other relevant information.
        config (RunnableConfig): The configuration for the instruction update process.
        store (BaseStore): The store where the instructions are saved.

    Returns:
        dict: A dictionary containing a message indicating the instruction update was successful.
    """
    config_manager = ConfigurationManager()
    chatbot_config = config_manager.get_chatbot_config()
    
    # Get the user ID from the config
    configurable = Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    # Define the namespace for the memories
    namespace = ("instructions", user_id)
    existing_memory = store.get(namespace, "user_instructions")
        
    # Format the memory in the system prompt
    system_msg = CREATE_INSTRUCTIONS.format(current_instructions=existing_memory.value if existing_memory else None)
    
    logger.info(f"Initiating instruction update for user ID \"{user_id}\"..")
    
    instruction_updater = InstructionUpdater(config=chatbot_config)
    new_memory = instruction_updater.update_instruction(system_msg, state["messages"])

    # Overwrite the existing memory in the store 
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    
    # Get the tool calls from the last message in the chat history
    tool_calls = state['messages'][-1].tool_calls
    
    logger.info(f"Instruction update for user ID \"{user_id}\" completed.")
    
    return {
        "messages": [{
            "role": "tool",
            "name": UPDATE_INSTRUCTIONS_TOOL,
            "content": "Instructions updated.", 
            "tool_call_id":tool_calls[0]['id']
        }]
    }