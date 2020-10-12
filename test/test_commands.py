import os
from os.path import dirname

import pytest

from tempocnn import version

entry_points = ['tempo', 'meter', 'tempogram', 'greekfolk']


@pytest.fixture
def test_track():
    dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dir, 'data', 'drumtrack.mp3')


@pytest.mark.parametrize('entry_point', entry_points)
def test_help(script_runner, entry_point):
    # sanity check -- can it be called at all
    ret = script_runner.run(entry_point, '--help')
    assert ret.success
    assert ret.stdout.startswith('usage:')
    assert ret.stderr == ''


@pytest.mark.parametrize('entry_point', entry_points)
def test_version(script_runner, entry_point):
    # sanity check -- can it be called at all
    ret = script_runner.run(entry_point, '--version')
    assert ret.success
    assert ret.stdout.startswith(entry_point)
    assert version.__version__ in ret.stdout
    assert ret.stderr == ''


def test_tempo(script_runner, test_track):
    ret = script_runner.run('tempo', '-i', test_track)
    assert ret.success
    assert 'Loading model' in ret.stdout
    assert '100' in ret.stdout


def test_tempogram(script_runner, test_track):
    ret = script_runner.run('tempogram', '-p', test_track)
    assert ret.success
    assert 'Loading model' in ret.stdout
    assert os.path.exists(test_track + '.png')


def test_meter(script_runner, test_track):
    ret = script_runner.run('meter', '-i', test_track)
    assert ret.success
    assert 'Loading model' in ret.stdout
    assert '9' in ret.stdout


def test_greekfolk(script_runner, tmpdir, test_track):
    ret = script_runner.run('greekfolk', dirname(test_track), str(tmpdir))
    assert ret.success
    assert 'No .wav files found in' in ret.stdout

# TODO: more comprehensive testing
