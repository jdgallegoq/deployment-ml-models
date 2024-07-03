import math
import numpy as np
from regresion_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # given
    expected_first_prediction_value = 113422
    expected_no_predictions = 1449

    # when
    result = make_prediction(input_data=sample_input_data)

    # then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=100)
