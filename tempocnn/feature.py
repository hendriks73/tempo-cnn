# encoding: utf-8

"""
Feature loading from audio files.

Specifically, tempo-cnn uses mel spectra with 40 bands ranging from
20 to 5000 Hz.
"""

import librosa as librosa
import numpy as np


def read_features(file, frames=256, hop_length=128, zero_pad=False):
    """
    Resample file to 11025 Hz, then transform using STFT with length 1024
    and hop size 512. Convert resulting linear spectrum to mel spectrum
    with 40 bands ranging from 20 to 5000 Hz.

    Since we require at least 256 frames, shorter audio excerpts are always
    zero padded.

    Specifically for tempogram 128 frames each can be added at the front and
    at the back in order to make the calculation of BPM values for the first
    and the last window possible.

    :param file: file
    :param frames: 256
    :param hop_length: 128 or shorter
    :param zero_pad: adds 128 zero frames both at the front and back
    :param normalize: normalization function
    :return: feature tensor for the whole file
    """
    y, sr = librosa.load(file, sr=11025)
    data = librosa.feature.melspectrogram(y=y, sr=11025, n_fft=1024, hop_length=512,
                                          power=1, n_mels=40, fmin=20, fmax=5000)
    data = np.reshape(data, (1, data.shape[0], data.shape[1], 1))

    # add frames/2 zero frames before and after the data
    if zero_pad:
        data = _add_zeros(data, frames)

    # zero-pad, if we have less than 256 frames to make sure we get some
    # result at all
    if data.shape[2] < frames:
        data = _ensure_length(data, frames)

    # convert data to overlapping windows,
    # each window is one sample (first dim)
    return _to_sliding_window(data, frames, hop_length)


def _ensure_length(data, length):
    padded_data = np.zeros((1, data.shape[1], length, 1), dtype=data.dtype)
    padded_data[0, :, 0:data.shape[2], 0] = data[0, :, :, 0]
    return padded_data


def _add_zeros(data, zeros):
    padded_data = np.zeros((1, data.shape[1], data.shape[2] + zeros, 1), dtype=data.dtype)
    padded_data[0, :, zeros // 2:data.shape[2] + (zeros // 2), 0] = data[0, :, :, 0]
    return padded_data


def _to_sliding_window(data, window_length, hop_length):
    total_frames = data.shape[2]
    windowed_data = []
    for offset in range(0, ((total_frames - window_length) // hop_length + 1) * hop_length, hop_length):
        windowed_data.append(np.copy(data[:, :, offset:window_length + offset, :]))
    return np.concatenate(windowed_data, axis=0)
