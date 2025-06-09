from MLProject.config.configuration import ConfigurationManager
from MLProject.components.preprocessing import Preprocessing
from MLProject import logger

STAGE_NAME = "Preprocessing"

class PreprocessingPipeline:
    def __init__(self):
        pass

    def pipeline(self):
        config = ConfigurationManager()
        preprocessing_config = config.get_preprocessing_config()
        preprocessing = Preprocessing(config=preprocessing_config)
        preprocessing.preprocess_data()

if __name__ == '__main__':
    try:
        logger.info(f"\n\n")
        logger.info(f">>>>>>> Stage {STAGE_NAME} Started <<<<<<<")
        
        obj = PreprocessingPipeline()
        obj.pipeline()
        
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e