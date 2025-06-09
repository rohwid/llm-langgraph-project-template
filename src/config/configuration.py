import os

from dataclasses import dataclass, fields
from langchain_core.runnables import RunnableConfig
from str_to_bool import str_to_bool
from typing import Any, Optional

from src.constants import CONFIG_FILE_PATH
from src.entity.config_entity import ChatbotConfig, APIConfig
from src.prompts import MODEL_SYSTEM_ROLE
from src.utils.common import read_yaml

"""
NOTE: Delete or replace any function as you need and don't forget to import each class config 
      from "../config/configuration.py" or "src/CISAIPrompter/config/configuration.py".
"""

class ConfigurationManager:
    """Manages the configuration of the application by reading and providing access to configuration settings."""
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        """Initialize ConfigurationManager with config and params files.
        
        Args:
            config_filepath: Path to config YAML file
        """
        self.config = read_yaml(config_filepath)
    
    def get_chatbot_config(self) -> ChatbotConfig:
        """Retrieves the ChatbotConfig object from the configuration file.
        
        This method constructs a ChatbotConfig object using settings from the configuration file and environment variables. 
        It does not create a download directory.
        
        Returns:
            A ChatbotConfig object containing model settings and Elasticsearch connection details.
        """
        chatbot_config = self.config.chatbot
        
        # Extract Elasticsearch environment variables
        es_host = os.environ["ES_HOST"]
        es_port = os.environ["ES_PORT"]
        
        # Construct and return the ChatbotConfig object
        config = ChatbotConfig(
            model=chatbot_config.model,
            temperature=chatbot_config.temperature,
            embedding_model=chatbot_config.embedding_model,
            is_macos=bool(str_to_bool(os.environ["IS_MACOS"])),
            es_url=f"http://{es_host}:{es_port}",
            es_index=os.environ["ES_INDEX"],
            content_field=chatbot_config.content_field,
            knn_field=chatbot_config.knn_field,
            k=chatbot_config.k,
            env=os.environ["ENV"]
        )

        return config
    
    def get_api_config(self) -> APIConfig:
        """Get ChatbotConfig with settings from config file.
        
        Creates download directory if it doesn't exist.
        
        Returns:
            ChatbotConfig object with download_dir and model settings
        """
        api_config = self.config.api
        
        # Create and return config object
        config = APIConfig(
            graph_name=api_config.graph_name,
            api_worker_numbers=int(os.environ["API_WORKER_NUMBERS"]),
            langgraph_server_url=os.environ["LANGGRAPH_SERVER_URL"],
            env=os.environ["ENV"]
        )

        return config

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    user_id: str = "default-user"
    model_system_role: str = MODEL_SYSTEM_ROLE

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        
        # Build dictionary of config values, 
        # checking environment variables first,
        # then falling back to configurable values
        values: dict[str, Any] = {}
        for field in fields(cls):
            if field.init:
                env_value = os.environ.get(field.name.upper())
                config_value = configurable.get(field.name)
                values[field.name] = env_value if env_value is not None else config_value
        
        return cls(**{k: v for k, v in values.items() if v})