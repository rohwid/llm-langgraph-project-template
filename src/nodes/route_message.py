from langgraph.graph import END
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

from src.states import QnAChatbotState
from typing import Literal

def route_message(state: QnAChatbotState, 
                  config: RunnableConfig, 
                  store: BaseStore) -> Literal[END, "update_instructions"]:
    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state["messages"][-1]
    if len(message.tool_calls) == 0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call["args"]["update_type"] == "INSTRUCTION":
            return "update_instructions"
        else:
            raise ValueError