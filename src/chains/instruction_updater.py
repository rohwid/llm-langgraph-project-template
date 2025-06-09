from src.entity.config_entity import ChatbotConfig

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

class InstructionUpdater:
    """
    This class is responsible for updating instructions based on the given system message and messages.
    It utilizes the ChatOpenAI model to process the messages and update the instructions accordingly.
    """
    def __init__(self, config: ChatbotConfig):
        """
        Initializes the InstructionUpdater with the given configuration.

        Args:
            config (PrompterConfig): The configuration for the InstructionUpdater.
        """
        self.config = config
        
    def update_instruction(self, system_msg: str, messages: object) -> object:
        """
        Updates the instructions based on the given system message and messages.

        This method utilizes the ChatOpenAI model to process the system message and messages, and then updates the instructions accordingly.
        It appends a human message to the end of the messages list, requesting the model to update the instructions based on the conversation.
        If an error occurs during the update process, it logs the error and raises the exception.

        Args:
            system_msg (str): The system message to be processed.
            messages (object): A list of messages to be processed.

        Returns:
            dict: The response from the model after processing the messages and updating the instructions.
        """
        model = ChatOpenAI(model=self.config.model, temperature=self.config.temperature)
        
        try:
            response = model.invoke(
                [SystemMessage(content=system_msg)] + messages[:-1] + \
                    [HumanMessage(content="Please update the instructions based on the conversation")]
            )
        except Exception as e:
            logger.error(f"Error to update instructions: {e}")
            raise
        
        return response