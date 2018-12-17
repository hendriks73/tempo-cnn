# encoding: utf-8

import sys
import os
import numpy as np
import pkgutil
import tempfile

from tensorflow.python.keras.models import load_model


class TempoClassifier:
    """
    Classifier that can estimate musical tempo in different formats.
    """

    def __init__(self, model_name='fcn'):
        """
        Initializes this classifier with a Keras model.

        :param model_name: model name from sub-package models. E.g. 'fcn', 'cnn', or 'ismir2018'
        """
        if 'fma' in model_name:
            # fma model uses log BPM scale
            factor = 256. / np.log(10)
            self._to_bpm = lambda index: np.exp((index + 435)/factor)
        else:
            self._to_bpm = lambda index: index + 30
        self.model_name = model_name
        resource = _to_model_resource(model_name)
        try:
            file = _extract_from_package(resource)
        except Exception as e:
            print('Failed to find a model named \'{}\'. Please check the model name.'.format(model_name),
                  file=sys.stderr)
            raise e
        try:
            self.model = load_model(file)
        finally:
            os.remove(file)

    def estimate(self, data):
        """
        Estimate a tempo distribution.
        Probabilities are indexed, starting with 30 BPM and ending with 286 BPM.

        :param data: features
        :return: tempo probability distribution
        """
        assert len(data.shape) == 4, 'Input data must be four dimensional. Actual shape was ' + str(data.shape)
        assert data.shape[1] == 40, 'Second dim of data must be 40. Actual shape was ' + str(data.shape)
        assert data.shape[2] == 256, 'Third dim of data must be 256. Actual shape was ' + str(data.shape)
        assert data.shape[3] == 1, 'Fourth dim of data must be 1. Actual shape was ' + str(data.shape)
        return self.model.predict(data, data.shape[0])

    def estimate_tempo(self, data):
        """
        Estimates the pre-dominant global tempo.

        :param data: features
        :return: a single tempo
        """
        prediction = self.estimate(data)
        averaged_prediction = np.average(prediction, axis=0)
        index = np.argmax(averaged_prediction)
        return self._to_bpm(index)

    def estimate_mirex(self, data):
        """
        Estimates the two dominant tempi along with a salience value.

        :param data: features
        :return: tempo1, tempo2, salience of tempo1
        """
        prediction = self.estimate(data)
        averaged_prediction = np.average(prediction, axis=0)
        peaks = self._find_bpm_peaks(averaged_prediction)

        if len(peaks) == 0:
            s1 = 1.
            t1 = 0.
            t2 = 0.
        elif len(peaks) == 1:
            bpm = peaks[0][0]
            if bpm > 120:
                alt = bpm/2
                s1 = 0.
                t1 = alt
                t2 = bpm
            else:
                alt = bpm*2
                s1 = 1.
                t1 = bpm
                t2 = alt
        else:
            bpm = peaks[0][0]
            bpm_height = peaks[0][1]
            alt = peaks[1][0]
            alt_height = peaks[1][1]
            if bpm < alt:
                s1 = bpm_height / (bpm_height+alt_height)
                t1 = bpm
                t2 = alt
            else:
                s1 = alt_height / (bpm_height+alt_height)
                t1 = alt
                t2 = bpm
        return t1, t2, s1

    def _find_bpm_peaks(self, distribution):
        peaks = []
        last_bpm = 0
        for index in range(256):
            bpm = self._to_bpm(index)
            height = distribution[index]
            start = max(index-5, 0)
            length = min(11, distribution.shape[0]-start)
            m = np.max(distribution[start:start + length])
            if height == m and bpm > last_bpm + 5:
                peaks.append((bpm, height))
                last_bpm = bpm
        # sort peaks by height, descending
        return sorted(peaks, key=lambda element: element[1], reverse=True)


class MeterClassifier:
    """
    Classifier that can estimate musical meter
    """

    def __init__(self, model_name='fcn'):
        """
        Initializes this classifier with a Keras model.

        :param model_name: model name from sub-package models. E.g. 'fma2018-meter'.
        """
        self._to_meter = lambda index: index + 2
        self.model_name = model_name
        resource = _to_model_resource(model_name)
        try:
            file = _extract_from_package(resource)
        except Exception as e:
            print('Failed to find a model named \'{}\'. Please check the model name.'.format(model_name),
                  file=sys.stderr)
            raise e
        try:
            self.model = load_model(file)
        finally:
            os.remove(file)

    def estimate(self, data):
        """
        Estimate a meter distribution.
        Probabilities are indexed, starting with 2. Only the meter numerator is given (e.g. 2 for 2/4).

        :param data: features
        :return: meter probability distribution
        """
        assert len(data.shape) == 4, 'Input data must be four dimensional. Actual shape was ' + str(data.shape)
        assert data.shape[1] == 40, 'Second dim of data must be 40. Actual shape was ' + str(data.shape)
        assert data.shape[2] == 512, 'Third dim of data must be 512. Actual shape was ' + str(data.shape)
        assert data.shape[3] == 1, 'Fourth dim of data must be 1. Actual shape was ' + str(data.shape)
        return self.model.predict(data, data.shape[0])

    def estimate_meter(self, data):
        """
        Estimates the pre-dominant global meter.

        :param data: features
        :return: a single meter (numerator)
        """
        prediction = self.estimate(data)
        averaged_prediction = np.average(prediction, axis=0)
        index = np.argmax(averaged_prediction)
        return self._to_meter(index)


def _to_model_resource(model_name):
    file = model_name
    if not model_name.endswith('.h5'):
        file = file + '.h5'
    if not file.startswith('models/'):
        file = 'models/' + file
    return file


def _extract_from_package(resource):
    data = pkgutil.get_data('tempocnn', resource)
    with tempfile.NamedTemporaryFile(prefix='model', suffix='.h5', delete=False) as f:
        f.write(data)
        name = f.name
    return name
