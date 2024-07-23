import math
import numpy as np
from regression_model.predict import make_prediction

def test_make_prediction(sample_input_data):
    # given
    expected_first_prediction_value = 0
    expected_no_predictions = 262
    # when
    result = make_prediction(input_data=sample_input_data)

    # then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], int)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert predictions[0] == expected_first_prediction_value