import unittest

import librosa

from tempocnn.feature import read_features


class TestTempoClassifier(unittest.TestCase):

    def test_init(self):
        file = 'data/drumtrack.mp3'
        y, sr = librosa.load(file, sr=11025)
        # possible features frames
        num_feature_frames = y.shape[0] / 512
        # possible feature windows with half overlap
        num_feature_windows = (num_feature_frames // 128) // 2

        features = read_features(file)
        self.assertEqual(len(features.shape), 4)
        self.assertEqual(features.shape[0], num_feature_windows)
        self.assertEqual(features.shape[1], 40)
        self.assertEqual(features.shape[2], 256)
        self.assertEqual(features.shape[3], 1)

