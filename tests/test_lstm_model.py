import pytest

pytest.importorskip("tensorflow")

from ml.train import create_lstm_model


def test_lstm_model_uses_sequence_input_shape():
    model = create_lstm_model()
    assert model.input_shape == (None, 30, 127)
    assert model.output_shape == (None, 10)
