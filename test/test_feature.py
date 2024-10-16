from pathlib import Path

import librosa
import pytest

from tempocnn.feature import read_features


@pytest.fixture
def test_track():
    return Path(__file__).absolute().parent / "data" / "drumtrack.mp3"


def test_init(test_track):
    y, sr = librosa.load(test_track, sr=11025)
    # possible features frames
    num_feature_frames = y.shape[0] / 512
    # possible feature windows with half overlap
    num_feature_windows = (num_feature_frames // 128) // 2

    features = read_features(test_track)
    assert len(features.shape) == 4
    assert features.shape[0] == num_feature_windows
    assert features.shape[1] == 40
    assert features.shape[2] == 256
    assert features.shape[3] == 1
