# import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tempocnn.classifier import (
    TempoClassifier,
    MeterClassifier,
    _load_model_from_github,
)
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
    return Path(__file__).absolute().parent / "data" / "drumtrack.mp3"


@pytest.mark.parametrize(
    "model_name",
    ["fcn", "cnn", "ismir2018", "mazurka", "deeptemp", "shallowtemp", "deepsquare"],
)
@patch("tempocnn.classifier.load_model")
def test_init(mock_load_model, model_name):
    mock_load_model.return_value = "model"
    tempo_classifier = TempoClassifier(model_name)
    assert tempo_classifier.model is not None
    mock_load_model.assert_called_once()


def test_bad_tempo_model_name():
    with pytest.raises(FileNotFoundError):
        TempoClassifier("bad_model_name")


def test_bad_meter_model_name():
    with pytest.raises(FileNotFoundError):
        MeterClassifier("bad_model_name")


def test_predict(test_data):
    tempo_classifier = TempoClassifier("fcn")
    predictions = tempo_classifier.estimate(test_data)
    assert predictions.shape == (2, 256)
    np.testing.assert_array_almost_equal(np.ones(2), np.sum(predictions, axis=1))
    tempi = np.argmax(predictions, axis=1) + 30
    assert tempi[0] == 163
    assert tempi[1] == 43


def test_predict_tempo(test_data):
    tempo_classifier = TempoClassifier("fcn")
    tempo = tempo_classifier.estimate_tempo(test_data)
    assert 43.0 == pytest.approx(tempo)


def test_predict_mirex(test_data):
    tempo_classifier = TempoClassifier("fcn")
    t1, t2, s1 = tempo_classifier.estimate_mirex(test_data)
    assert s1 == pytest.approx(0.7373429, abs=0.001)
    assert t1 == pytest.approx(43.0)
    assert t2 == pytest.approx(86.0)


def test_mirex_tempo_sanity(test_data):
    tempo_classifier = TempoClassifier("fcn")
    t1, t2, s1 = tempo_classifier.estimate_mirex(test_data)
    tempo = tempo_classifier.estimate_tempo(test_data)
    if s1 > 0.5:
        assert tempo == t1
    else:
        assert tempo == t2


def test_mirex_with_real_data(test_track):
    features = read_features(test_track)
    tempo_classifier = TempoClassifier("fcn")
    t1, t2, s1 = tempo_classifier.estimate_mirex(features)
    assert s1 == pytest.approx(0.99617153, abs=0.01)
    assert t1 == pytest.approx(100.0)
    assert t2 == pytest.approx(201.0)


def test_mirex_with_real_data_interpolate(test_track):
    features = read_features(test_track)
    tempo_classifier = TempoClassifier("fcn")
    t1, t2, s1 = tempo_classifier.estimate_mirex(features, interpolate=True)
    assert s1 == pytest.approx(0.99617153, abs=0.01)
    assert t1 == pytest.approx(100.000, abs=0.001)
    assert t2 == pytest.approx(200.984, abs=0.001)


@pytest.mark.parametrize(
    "model_name,tempo",
    [
        ("cnn", 100.0),
        ("deepsquare_k1", 102.0),
        ("deepsquare_k2", 100.0),
        ("deepsquare_k4", 100.0),
        ("deepsquare_k8", 100.0),
        ("deepsquare_k16", 100.0),
        ("deepsquare_k24", 100.0),
        ("deeptemp_k2", 100.0),
        ("deeptemp_k4", 100.0),
        ("deeptemp_k8", 100.0),
        ("deeptemp_k16", 100.0),
        ("deeptemp_k24", 100.0),
        # dt_maz models are not meant for drumtracks...
        ("dt_maz_m_fold0", 160.0),
        ("dt_maz_m_fold1", 93.0),
        ("dt_maz_m_fold2", 73.0),
        ("dt_maz_m_fold3", 156.0),
        ("dt_maz_m_fold4", 83.0),
        ("dt_maz_v_fold0", 160.0),
        ("dt_maz_v_fold1", 93.0),
        ("dt_maz_v_fold2", 73.0),
        ("dt_maz_v_fold3", 156.0),
        ("dt_maz_v_fold4", 83.0),
        ("fcn", 100.0),
        ("fma2018", 100.0),
        ("ismir2018", 100.0),
        ("shallowtemp_k1", 100.0),
        ("shallowtemp_k2", 100.0),
        ("shallowtemp_k4", 100.0),
        ("shallowtemp_k6", 100.0),
        ("shallowtemp_k8", 100.0),
        ("shallowtemp_k12", 100.0),
        ("deeptemp", 100.0),
        ("shallowtemp", 100.0),
        ("deepsquare", 100.0),
    ],
)
def test_tempo_with_real_data(model_name, tempo, test_track):
    features = read_features(test_track)
    tempo_classifier = TempoClassifier(model_name)
    tempo = tempo_classifier.estimate_tempo(features)
    assert tempo == pytest.approx(tempo)


def test_quad_interpol_argmax():
    index, strength = TempoClassifier.quad_interpol_argmax(np.array([0, 2, 0]))
    assert index == pytest.approx(1.0)
    assert strength == pytest.approx(2.0)


def test_quad_interpol_argmax_x_zero():
    index, strength = TempoClassifier.quad_interpol_argmax(np.array([0, 2, 0]), x=0)
    assert index == pytest.approx(0.0)
    assert strength == pytest.approx(0.0)


def test_quad_interpol_argmax_x_non_zero():
    index, strength = TempoClassifier.quad_interpol_argmax(np.array([0, 2, 0]), x=1)
    assert index == pytest.approx(1.0)
    assert strength == pytest.approx(2.0)


def test_load_model_from_github():
    # smallest model
    file = _load_model_from_github("models/deepsquare_k1.h5")
    assert file is not None
