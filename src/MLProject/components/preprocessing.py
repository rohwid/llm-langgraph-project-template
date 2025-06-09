from MLProject import logger
from MLProject.entity.config_entity import PreprocessingConfig

class Preprocessing:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def preprocess_data(self):
        """dump the splited dataset into data training and testing.
        you can also preprocess the data in this component
        
        THIS ONLY TEMPLATE!
        so, we just print the variables
        """
        try:
            logger.info("THIS JUST TEMPLATE! So, we will only print variables")
            logger.info(f"Data preprocessing root directory at {self.config.root_dir}")
            logger.info(f"Dataset train in {self.config.input_train_path}")
            logger.info(f"Dataset test in {self.config.input_test_path}")
            logger.info(f"Preprocessed data train in {self.config.output_train_path}")
            logger.info(f"Preprocessed data test in {self.config.output_test_path}")
        except Exception as e:
            logger.error(e)
            raise e
