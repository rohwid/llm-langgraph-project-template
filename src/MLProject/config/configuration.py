import os

from pathlib import Path
from MLProject.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from MLProject.utils.common import read_yaml, create_directories
from MLProject.entity.config_entity import (DataIngestionConfig, 
                                            PreprocessingConfig,
                                            TrainingConfig,
                                            TrainEvaluationConfig)

"""NOTE: Delete or replace any function as you need
and don't forget to import each class config from
'../config/configuration.py' or
'src/MLProject/config/configuration.py'
"""

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([Path(self.config.artifacts_root)])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """read data ingestion config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: DataIngestionConfig type
        """
        data_ingestion_config = self.config.data_ingestion

        create_directories([Path(data_ingestion_config.root_dir)])

        config = DataIngestionConfig(
            root_dir=Path(data_ingestion_config.root_dir),
            input_train_path=Path(data_ingestion_config.input_train_path),
            input_test_path=data_ingestion_config.input_test_path,
        )

        return config
    
    def get_preprocessing_config(self) -> PreprocessingConfig:
        """read preprocessing config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: PreprocessingConfig type
        """
        data_ingestion_config = self.config.data_ingestion
        preprocessing_config = self.config.preprocessing

        create_directories([Path(preprocessing_config.root_dir)])

        config = PreprocessingConfig(
            root_dir=Path(data_ingestion_config.root_dir),
            input_train_path=Path(data_ingestion_config.input_train_path),
            input_test_path=Path(data_ingestion_config.input_test_path),
            output_train_path=Path(preprocessing_config.output_train_path),
            output_test_path=Path(preprocessing_config.output_test_path),
        )

        return config
    
    def get_training_config(self) -> TrainingConfig:
        """read training config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: TrainingConfig type
        """
        data_ingestion_config = self.config.data_ingestion
        preprocessing_config = self.config.preprocessing
        training_config = self.config.train_model
        train_params = self.params

        create_directories([Path(training_config.root_dir)])

        config = TrainingConfig(
            root_dir=Path(data_ingestion_config.root_dir),
            input_train_path=Path(data_ingestion_config.input_train_path),
            input_test_path=Path(data_ingestion_config.input_test_path),
            output_train_path=Path(preprocessing_config.output_train_path),
            output_test_path=Path(preprocessing_config.output_test_path),
            model_path=Path(training_config.model_path),
            params_batch_size=train_params.BATCH_SIZE,
            params_epoch=train_params.EPOCH,
            params_classes=train_params.CLASSES,
            params_lr=train_params.LEARNING_RATE
        )

        return config
    
    def get_train_eval_config(self) -> TrainEvaluationConfig:
        """read evaluation config file and store as config entity
        then apply the dataclasses
        
        Returns:
            config: TrainEvaluationConfig type
        """
        data_ingestion_config = self.config.data_ingestion
        preprocessing_config = self.config.preprocessing
        training_config = self.config.train_model
        evaluation_config = self.config.evaluation
        
        create_directories([Path(evaluation_config.root_dir)])

        config = TrainEvaluationConfig(
            root_dir=Path(data_ingestion_config.root_dir),
            input_train_path=Path(data_ingestion_config.input_train_path),
            input_test_path=Path(data_ingestion_config.input_test_path),
            output_train_path=Path(preprocessing_config.output_train_path),
            output_test_path=Path(preprocessing_config.output_test_path),
            model_path=Path(training_config.model_path),
            score_path=Path(evaluation_config.score_path),
            mlflow_dataset_path=Path(evaluation_config.mlflow_dataset_path),
            mlflow_tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            mlflow_exp_name=evaluation_config.mlflow_exp_name,
            mlflow_run_name=evaluation_config.mlflow_run_name
        )

        return config