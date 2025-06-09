from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from loguru import logger
from typing import Any, Dict, List

from src.chains.memory_updater import MemoryUpdater
from src.config.configuration import Configuration, ConfigurationManager
from src.prompts import MODEL_SYSTEM_MESSAGE
from src.states import QnAChatbotState

def metta_chatbot(state: QnAChatbotState, config: RunnableConfig, store: BaseStore) -> Dict[str, List[Any]]:
    """Loads memories from the store and uses them to personalize the chatbot's response."""
    config_manager = ConfigurationManager()
    prompter_config = config_manager.get_chatbot_config()
    
    # Extracts the user ID and model system role from the config
    configurable = Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    # Enable or disable debug
    DEBUG = False
    
    logger.info(f"Updating memory for user ID \"{user_id}\"..")
    
    model_system_role = configurable.model_system_role

    # Retrieves custom instructions from the store
    logger.info(f"Retrieving instructions for user ID \"{user_id}\"..")
    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    instructions = memories[0].value if memories else None
    logger.info(f"Retrieving instructions for user ID \"{user_id}\".. done")
    
    # Extracts the context from the state
    context = state["retrieved_docs"]
    
    logger.info(context)
    
    # Formats the system message with the model system role, context, and instructions
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        model_system_role=model_system_role,
        context=context,
        instructions=instructions
    )
    
    logger.info(prompter_config.env)
    
    if DEBUG and prompter_config.env == "development":
        logger.info(f"Checking messages for \"{user_id}\"..")
        logger.info(f"current system_msg: {system_msg}")
        logger.info(f"state messages: {state["messages"]}")
        logger.info(f"Checking messages for \"{user_id}\".. done")

    # Updates the memory and generates a response based on the system message and state messages
    memory_updater = MemoryUpdater(config=prompter_config)
    response = memory_updater.update_memory(system_msg, state["messages"])
    
    logger.info(f"Updating memory for user ID \"{user_id}\".. done")

    # Returns a dictionary containing the response as a list of messages
    return {
        "messages": [response]
    }