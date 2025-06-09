from dataclasses import dataclass
from pydantic import Field
from typing import Literal

"""NOTE: Delete or replace any class as you need and don't forget to import this class in
"../config/configuration.py" or "src/CISAIPrompter/config/configuration.py"
"""

@dataclass(frozen=True)
class ChatbotConfig:
    model: str = Field(..., description="The model to use for the chatbot.")
    temperature: int = Field(..., description="The temperature for the model's output.")
    embedding_model: str = Field(..., description="The embedding model to use for the chatbot.")
    is_macos: str = Field(..., description="Indicates if the system is macOS.")
    es_url: str = Field(..., description="The URL for the Elasticsearch instance.")
    es_index: str = Field(..., description="The index name in Elasticsearch.")
    content_field: str = Field(..., description="The field name for content in Elasticsearch.")
    knn_field: str = Field(..., description="The field name for KNN in Elasticsearch.")
    k: int = Field(..., description="The number of nearest neighbors to consider.")
    env: Literal["development", "staging", "production"] = Field(..., description="The environment in which the chatbot is running.")

@dataclass(frozen=True)
class APIConfig:
    graph_name: str = Field(..., description="The name of the graph to use.")
    api_worker_numbers: int = Field(..., description="The number of API worker processes to run.")
    langgraph_server_url: str = Field(..., description="The URL of the LangGraph server.")
    env: Literal["development", "staging", "production"] = Field(..., description="The environment in which the API is running.")    