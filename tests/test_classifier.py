import unittest

import numpy as np

from tempocnn.classifier import TempoClassifier
from tempocnn.feature import read_features


class TestTempoClassifier(unittest.TestCase):

    def test_init(self):
        tempo_classifier = TempoClassifier('fcn')
        self.assertIsNotNone(tempo_classifier.model)

    def test_bad_model_name(self):
        try:
            TempoClassifier('bad_model_name')
            self.fail('Expected FileNotFoundError')
        except FileNotFoundError:
            pass

    def test_predict(self):
        tempo_classifier = TempoClassifier('fcn')
        predictions = tempo_classifier.estimate(self.get_test_data())
        self.assertEqual(predictions.shape, (2, 256))
        np.testing.assert_array_almost_equal(np.ones(2), np.sum(predictions, axis=1))
        tempi = np.argmax(predictions, axis=1) + 30
        self.assertEqual(tempi[0], 163)
        self.assertEqual(tempi[1], 43)

    def test_predict_tempo(self):
        tempo_classifier = TempoClassifier('fcn')
        tempo = tempo_classifier.estimate_tempo(self.get_test_data())
        self.assertAlmostEqual(43., tempo)

    def test_predict_mirex(self):
        tempo_classifier = TempoClassifier('fcn')
        t1, t2, s1 = tempo_classifier.estimate_mirex(self.get_test_data())
        self.assertAlmostEquals(s1, 0.7373429)
        self.assertAlmostEquals(t1, 43.)
        self.assertAlmostEquals(t2, 86.)

    def test_mirex_tempo_sanity(self):
        tempo_classifier = TempoClassifier('fcn')
        t1, t2, s1 = tempo_classifier.estimate_mirex(self.get_test_data())
        tempo = tempo_classifier.estimate_tempo(self.get_test_data())
        if s1 > 0.5:
            self.assertEqual(tempo, t1)
        else:
            self.assertEqual(tempo, t2)

    def test_mirex_with_real_data(self):
        features = read_features('data/drumtrack.mp3')
        tempo_classifier = TempoClassifier('fcn')
        t1, t2, s1 = tempo_classifier.estimate_mirex(features)
        self.assertAlmostEquals(s1, 0.99617153)
        self.assertAlmostEquals(t1, 100.)
        self.assertAlmostEquals(t2, 101.)

    def test_tempo_with_real_data(self):
        features = read_features('data/drumtrack.mp3')
        tempo_classifier = TempoClassifier('fcn')
        tempo = tempo_classifier.estimate_tempo(features)
        self.assertAlmostEqual(100., tempo)

    def get_test_data(self):
        artificial_data = np.zeros((2, 40, 256, 1))
        for i in np.arange(0, 256, 4):
            artificial_data[0, :, i, 0] = 1
        for i in np.arange(0, 256, 30):
            artificial_data[1, :, i, 0] = 1
        return artificial_data
