from pathlib import Path
from typing import Dict, List, Optional, Sequence
from pydantic import BaseModel
from strictyaml import YAML, load
import regression_model

# project directories
PACKAGE_ROOT = Path(regression_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    App level config
    """
    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str

class ModelConfig(BaseModel):
    """
    All config relevant to model training
    and feature engineering
    """
    # target variable name
    target: str
    # all features list must be strings
    features: List[str]
    # for model training
    test_size: float
    random_state: int
    alpha: float
    # for feature engineering
    title_variable: str
    to_float_variables: List[str]
    drop_variables: List[str]
    numerical_variables: List[str]
    categorical_variables: List[str]
    cabin: str
    
    
class Config(BaseModel):
    """
    Master config object
    """
    app_config: AppConfig
    model_config: ModelConfig

def find_config_file() -> Path:
    """
    Locate the config file
    """
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"config file not found at {CONFIG_FILE_PATH}")

def fetch_config_from_yaml(cfg_path: Optional[Path]=None) -> YAML:
    """
    Parse YAML containing the package config
    """
    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as config_file:
            parsed_config = load(config_file.read())
            return parsed_config
    
    raise OSError(f"Did not find config file at path {cfg_path}")

def create_and_validate_config(parsed_config: YAML=None)-> Config:
    """
    Run and validates on config values
    """
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify data attributes from the strictyaml YAML type
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data)
    )

    return _config

config = create_and_validate_config()