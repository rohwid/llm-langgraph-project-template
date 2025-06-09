from MLProject import logger
from MLProject.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def model(self):
        """perform training or train the data train
        
        THIS ONLY TEMPLATE!
        so, we just print the variables
        """
        try:
            logger.info("THIS JUST TEMPLATE! So, we will only print variables")
            logger.info(f"Training root directory at {self.config.root_dir}")
            logger.info(f"Dataset train in {self.config.input_train_path}")
            logger.info(f"Dataset test in {self.config.input_test_path}")
            logger.info(f"Preprocessed data train in {self.config.output_train_path}")
            logger.info(f"Preprocessed data test in {self.config.output_test_path}")
            logger.info(f"The model path in {self.config.model_path}")
            logger.info(f"BATCH SIZE: {self.config.params_batch_size}")
            logger.info(f"EPOCH: {self.config.params_epoch}")
            logger.info(f"CLASSES: {self.config.params_classes}")
            logger.info(f"LR: {self.config.params_lr}")
        except Exception as e:
            logger.error(e)
            raise e