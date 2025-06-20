from src.chains.sbe_retriever import RetrieverTool
from src.config.configuration import ConfigurationManager
from src.states import QnAChatbotState

from typing import Dict, List

def retrieve(state: QnAChatbotState) -> List[Dict[str, str]]:
    """
    Retrieves documents based on the given state and configuration.

    This function uses the RetrieverTool to fetch documents based on the query, domain URL, title IDs, and Elasticsearch configuration.
    It returns a dictionary containing the retrieved documents.

    Args:
        state (QnAChatbotState): The current state of the chatbot, including messages, domain URL, and title IDs.

    Returns:
        List[Dict[str, str]]: A dictionary containing the retrieved documents.
    """
    config_manager = ConfigurationManager()
    chatbot_config = config_manager.get_chatbot_config()
    retriever = RetrieverTool()
    
    retrieved_docs = retriever.invoke({
        "query": state["messages"][-1].content, # state["messages"] keep store all user messages
        "is_macos": chatbot_config.is_macos,
        "es_url": chatbot_config.es_url,
        "es_index": chatbot_config.es_index,
        "content_field": chatbot_config.content_field,
        "knn_field": chatbot_config.knn_field,
        "k": chatbot_config.k
    })
    
    return {
        "retrieved_docs": retrieved_docs
    }