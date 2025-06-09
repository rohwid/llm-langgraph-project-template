from MLProject.config.configuration import ConfigurationManager
from MLProject.components.model_evaluation import TrainEvaluation
from MLProject import logger

STAGE_NAME = "Training Evaluation"

class TrainEvaluationPipeline:
    def __init__(self):
        pass

    def pipeline(self):
        config = ConfigurationManager()
        train_eval_config = config.get_train_eval_config()
        train_eval = TrainEvaluation(config=train_eval_config)
        train_eval.mlflow_log_train()

if __name__ == '__main__':
    try:
        logger.info(f"\n\n")
        logger.info(f">>>>>>> Stage {STAGE_NAME} Started <<<<<<<")
        
        obj = TrainEvaluationPipeline()
        obj.pipeline()
        
        logger.info(f">>>>>> Stage {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
