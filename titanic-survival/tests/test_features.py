from regression_model.config.core import config
from regression_model.processing.features import ExtractLetterTransformer

def test_extract_letter_transformer(sample_input_data):
    # given
    transformer = ExtractLetterTransformer(
        variable=config.model_config.cabin
    )
    assert sample_input_data['cabin'][5]=='G6'


    # when
    subject = transformer.fit_transform(sample_input_data)
    # then
    assert subject["cabin_letter"][5]=='G'
