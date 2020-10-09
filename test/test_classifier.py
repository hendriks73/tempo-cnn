import os

import numpy as np
import pytest

from tempocnn.classifier import TempoClassifier
from tempocnn.feature import read_features


@pytest.fixture
def test_data():
    artificial_data = np.zeros((2, 40, 256, 1))
    for i in np.arange(0, 256, 4):
        artificial_data[0, :, i, 0] = 1
    for i in np.arange(0, 256, 30):
        artificial_data[1, :, i, 0] = 1
    return artificial_data


@pytest.fixture
def test_track():
    dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dir, 'data', 'drumtrack.mp3')


def test_init():
    tempo_classifier = TempoClassifier('fcn')
    assert tempo_classifier.model is not None


def test_bad_model_name():
    with pytest.raises(FileNotFoundError):
        TempoClassifier('bad_model_name')


def test_predict(test_data):
    tempo_classifier = TempoClassifier('fcn')
    predictions = tempo_classifier.estimate(test_data)
    assert predictions.shape == (2, 256)
    np.testing.assert_array_almost_equal(np.ones(2), np.sum(predictions, axis=1))
    tempi = np.argmax(predictions, axis=1) + 30
    assert tempi[0] == 163
    assert tempi[1] == 43


def test_predict_tempo(test_data):
    tempo_classifier = TempoClassifier('fcn')
    tempo = tempo_classifier.estimate_tempo(test_data)
    assert 43. == pytest.approx(tempo)


def test_predict_mirex(test_data):
    tempo_classifier = TempoClassifier('fcn')
    t1, t2, s1 = tempo_classifier.estimate_mirex(test_data)
    assert s1 == pytest.approx(0.7373429, abs=0.001)
    assert t1 == pytest.approx(43.)
    assert t2 == pytest.approx(86.)


def test_mirex_tempo_sanity(test_data):
    tempo_classifier = TempoClassifier('fcn')
    t1, t2, s1 = tempo_classifier.estimate_mirex(test_data)
    tempo = tempo_classifier.estimate_tempo(test_data)
    if s1 > 0.5:
        assert tempo == t1
    else:
        assert tempo == t2


def test_mirex_with_real_data(test_track):
    features = read_features(test_track)
    tempo_classifier = TempoClassifier('fcn')
    t1, t2, s1 = tempo_classifier.estimate_mirex(features)
    assert s1 == pytest.approx(0.99617153, abs=0.01)
    assert t1 == pytest.approx(100.)
    assert t2 == pytest.approx(201.)


def test_tempo_with_real_data(test_track):
    features = read_features(test_track)
    tempo_classifier = TempoClassifier('fcn')
    tempo = tempo_classifier.estimate_tempo(features)
    assert tempo == pytest.approx(100.)
