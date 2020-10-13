import os
from os.path import dirname
import jams

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


def test_tempo_interpolate(script_runner, test_track):
    ret = script_runner.run('tempo', '--interpolate', '-i', test_track)
    assert ret.success
    assert 'Loading model' in ret.stdout
    assert '100' in ret.stdout


def test_tempo_model(script_runner, test_track):
    ret = script_runner.run('tempo', '-m', 'deeptemp', '-i', test_track)
    assert ret.success
    assert 'Loading model' in ret.stdout
    assert '100' in ret.stdout


def test_tempo_mirex(script_runner, test_track):
    ret = script_runner.run('tempo', '--mirex', '-i', test_track)
    assert ret.success
    assert 'Loading model' in ret.stdout
    assert '100	201	0.99' in ret.stdout


def test_tempo_jams(script_runner, test_track):
    ret = script_runner.run('tempo', '--jams', '-i', test_track)
    assert ret.success
    assert 'Loading model' in ret.stdout
    assert 'Processing file' in ret.stdout
    jams_file = test_track.replace('.mp3', '.jams')
    assert os.path.exists(jams_file)
    jam = jams.load(jams_file)

    annotation = jam.annotations[0]
    assert annotation.duration == pytest.approx(15.046, abs=0.001)

    observation0 = annotation.data[0]
    assert observation0.time == 0.0
    assert observation0.confidence == pytest.approx(0.9979, abs=0.0001)
    assert observation0.value == 100

    observation1 = annotation.data[1]
    assert observation1.time == 0.0
    assert observation1.confidence == pytest.approx(0.0020, abs=0.0001)
    assert observation1.value == 201
    # TODO: check JAMS metadata


def test_tempo_jams_and_mirex(script_runner, test_track):
    ret = script_runner.run('tempo', '--jams', '--mirex', '-i', test_track)
    assert not ret.success


def test_tempo_extension(script_runner, test_track):
    ret = script_runner.run('tempo', '-e', '.fancy_pants', '-i', test_track)
    assert ret.success
    extension_name = test_track + '.fancy_pants'
    assert os.path.exists(extension_name)
    with open(extension_name, 'r') as f:
        assert '100' in f.read()


def test_tempo_replace_extension(script_runner, test_track):
    ret = script_runner.run('tempo', '-re', '.fancy_pants', '-i', test_track)
    assert ret.success
    extension_name = test_track.replace('.mp3', '.fancy_pants')
    assert os.path.exists(extension_name)
    with open(extension_name, 'r') as f:
        assert '100' in f.read()


def test_tempo_replace_exclusive(script_runner, test_track):
    ret = script_runner.run('tempo', '-e', '.fancy_pants', '-re', '.fancy_pants2', '-i', test_track)
    assert not ret.success


def test_tempo_cont(script_runner, test_track):
    ret = script_runner.run('tempo', '--cont', '-i', test_track)
    assert ret.success

# TODO: more comprehensive testing
