from box.exceptions import BoxValueError
from box import ConfigBox
from ensure import ensure_annotations
from loguru import logger
from pathlib import Path
from typing import Any

import os
import base64
import json
import joblib
import yaml

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
    except BoxValueError:
        logger.error("YAML file: The file is empty.")
        raise ValueError("YAML file: The file is empty.")
    except Exception as e:
        logger.error(f"YAML file: Error while reading the file: {e}")
        raise
    finally:
        logger.info(f"YAML file: {path_to_yaml} loaded successfully.")
        
    return ConfigBox(content)
    
@ensure_annotations
def create_directories(path_to_directories: list):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)
        logger.info(f"Json file loaded succesfully from: {path}")
    
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    
    return f"~ {size_in_kb} KB"

@ensure_annotations
def decode_image(img_str: str, file_name: Path):
    img_data = base64.b64decode(img_str)
    with open(file_name, 'wb') as f:
        f.write(img_data)
        f.close()
        
@ensure_annotations
def encode_image_base64(cropped_imag_path: Path):
    with open(cropped_imag_path, "rb") as f:
        return base64.b64encode(f.read())