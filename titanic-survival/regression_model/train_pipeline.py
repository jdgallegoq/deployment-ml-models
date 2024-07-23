import numpy as np
from config.core import config
from pipeline import survived_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split

def run_training() -> None:
    """
    Train model
    """
    # read training dataset
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide data
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state
    )
    # fit model
    survived_pipe.fit(X_train, y_train)
    # persis to save model
    save_pipeline(pipeline_to_persist=survived_pipe)

if __name__=='__main__':
    run_training()