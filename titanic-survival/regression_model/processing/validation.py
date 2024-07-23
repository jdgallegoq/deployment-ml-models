from typing import List, Optional, Tuple

import re
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from regression_model.config.core import config

def validate_inputs(*, input_data:pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Check model inputs for unprocesable values
    """

    relevant_data = input_data[config.model_config.features].copy()
    relevant_data = relevant_data.replace('?', np.nan)
    relevant_data[config.model_config.to_float_variables] = relevant_data[config.model_config.to_float_variables].astype(float)
    relevant_data.drop(config.model_config.drop_variables, inplace=True)
    # get first cabin
    relevant_data[config.model_config.cabin] = relevant_data[config.model_config.cabin].apply(get_first_cabin)
    # get title from name
    relevant_data[config.model_config.title_variable] = relevant_data[config.model_config.title_variable].apply(get_title)
    errors = None

    try:
        MultipleTitanicDataInputs(
            inputs=relevant_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()
    
    return relevant_data, errors

class TitanicDataInputSchema(BaseModel):
    pclass: Optional[str]
    sex: Optional[str]
    age: Optional[float]
    sibsp: Optional[int]
    parch: Optional[int]
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]
    title: Optional[str]
    #cabin_letter: Optional[str] # calculated

class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]


def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan
    
def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'