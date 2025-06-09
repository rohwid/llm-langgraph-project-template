from MLProject import logger
from MLProject.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest_data(self):
        """get data from source
        
        THIS JUST TEMPLATE!
        so, we just print the variables
        """
        try: 
            logger.info("THIS JUST TEMPLATE! So, we will only print variables")
            logger.info(f"Data ingestion root directory at {self.config.root_dir}")
            logger.info(f"Dataset train at {self.config.input_train_path}")
            logger.info(f"Dataset test at {self.config.input_test_path}")
        except Exception as e:
            logger.error(e)
            raise e
