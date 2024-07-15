from regresion_model.config.core import config
from regresion_model.processing.features import TemporalVariableTransformer


def test_temporal_variable_transformer(sample_input_data):
    # given
    transformer = TemporalVariableTransformer(
        variables=config.model_config.temporal_vars,
        reference_variable=config.model_config.ref_var,
    )
    assert sample_input_data["YearRemodAdd"].iat[0] == 1961

    # when
    subject = transformer.fit_transform(sample_input_data)
    # then
    assert subject["YearRemodAdd"].iat[0] == 49
