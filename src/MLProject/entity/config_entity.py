from dataclasses import dataclass
from pathlib import Path

"""NOTE: Delete or replace any class as you need
and don't forget to import this class in
'../config/configuration.py' or 
'src/MLProject/config/configuration.py'
"""

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    input_train_path: Path
    input_test_path: Path
    
@dataclass(frozen=True)
class PreprocessingConfig:
    root_dir: Path
    input_train_path: Path
    input_test_path: Path
    output_train_path: Path
    output_test_path: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    input_train_path: Path
    input_test_path: Path
    output_train_path: Path
    output_test_path: Path
    model_path: Path
    params_batch_size: int
    params_epoch: int
    params_classes: int
    params_lr: float

@dataclass(frozen=True)
class TrainEvaluationConfig:
    root_dir: Path
    input_train_path: Path
    input_test_path: Path
    output_train_path: Path
    output_test_path: Path
    model_path: Path
    score_path: Path
    mlflow_dataset_path: Path
    mlflow_tracking_uri: str
    mlflow_exp_name: str
    mlflow_run_name: str