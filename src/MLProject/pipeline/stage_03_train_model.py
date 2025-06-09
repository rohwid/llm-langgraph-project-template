from MLProject.config.configuration import ConfigurationManager
from MLProject.components.training import Training
from MLProject import logger

STAGE_NAME = "Training"

class TrainingPipeline:
    def __init__(self):
        pass

    def pipeline(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.model()
if __name__ == '__main__':
    try:
        logger.info(f"\n\n")
        logger.info(f">>>>>>> Stage {STAGE_NAME} Started <<<<<<<")
        
        obj = TrainingPipeline()
        obj.pipeline()
        
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
