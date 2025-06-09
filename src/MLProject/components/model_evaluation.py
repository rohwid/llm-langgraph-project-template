from MLProject import logger
from MLProject.entity.config_entity import TrainEvaluationConfig

class TrainEvaluation:
    def __init__(self, config: TrainEvaluationConfig):
        self.config = config

    def mlflow_log_train(self) -> None:
        """perform training or train the data train
        
        THIS ONLY TEMPLATE!
        so, we just print the variables
        """
        try:
            logger.info("THIS JUST TEMPLATE! So, we will only print variables")
            logger.info(f"Training root directory at {self.config.root_dir}")
            logger.info(f"Dataset train in {self.config.input_train_path}")
            logger.info(f"Dataset test in {self.config.input_test_path}")
            logger.info(f"Preprocess data train in {self.config.output_train_path}")
            logger.info(f"Preprocess data test in {self.config.output_test_path}")
            logger.info(f"The model path in {self.config.model_path}")
            logger.info(f"The MLflow tracking URI: {self.config.mlflow_tracking_uri}")
            logger.info(f"The MLflow experiment name: {self.config.mlflow_exp_name}")
            logger.info(f"The MLflow run name: {self.config.mlflow_run_name}")
        except Exception as e:
            logger.error(e)
            raise e