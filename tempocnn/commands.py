import argparse
import sys
from os import listdir, makedirs
from os.path import isfile, join, basename, exists, splitext, dirname

import jams
import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from tempocnn.classifier import TempoClassifier, MeterClassifier
from tempocnn.feature import read_features
from tempocnn.version import __version__ as package_version


def tempo():
    """tempo command line entry point."""

    # define parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The program 'tempo' estimates a global tempo for a given file.
    The underlying algorithm and the model ismir2018 is described in detail in:

    Hendrik Schreiber, Meinard Müller,
    "A single-step approach to musical meter estimation using a
    convolutional neural network"
    Proceedings of the 19th International Society for Music Information
    Retrieval Conference (ISMIR), Paris, France, Sept. 2018.

    Models fcn and cnn are from:

    Hendrik Schreiber,
    "CNN-based tempo estimation"
    Music Information Retrieval Evaluation eXchange (MIREX),
    Paris, France, 2018.

    Model fma2018 is from:

    Hendrik Schreiber,
    "Technical Report: Tempo and Meter Estimation for Greek Folk Music Using
    Convolutional Neural Networks and Transfer Learning"
    8th International Workshop on Folk Music Analysis (FMA),
    Thessaloniki, Greece, June 2018.

    Models deeptemp*, deepsquare* and shallowtemp* are from:

    Hendrik Schreiber, Meinard Müller,
    "Musical Tempo and Key Estimation using Convolutional
    Neural Networks with Directional Filters"
    Sound and Music Computing Conference (SMC),
    Málaga, Spain, 2019.

    Models mazurka, dt_maz_* are from:

    Hendrik Schreiber, Frank Zalkow, Meinard Müller,
    "Modeling and Estimating Local Tempo: A Case Study on Chopin’s Mazurkas"
    Proceedings of the 21st International Society for Music Information
    Retrieval Conference (ISMIR), Montréal, QC, Canada, Oct. 2020.

    License: GNU Affero General Public License v3
    ''')

    parser.add_argument('-v', '--version',
                        action='version',
                        version=f'tempo {package_version}')
    parser.add_argument('--interpolate',
                        help='interpolate between tempo classes',
                        action='store_true')
    parser.add_argument('-m', '--model',
                        nargs='?',
                        default='fcn',
                        help='model name [ismir2018|fma2018|cnn|fcn|mazurka|deeptemp|deepsquare|shallowtemp], '
                             'defaults to "fcn." For more sepcific model names, please check the repo.')
    parser.add_argument('-c', '--cont',
                        help='continue after error, if multiple files are processed',
                        action='store_true')

    extensions = parser.add_mutually_exclusive_group()
    extensions.add_argument('-e', '--extension',
                            help='append given extension to original file name for results')
    extensions.add_argument('-re', '--replace_extension',
                            help='replace the file existing with the given one')

    output_format = parser.add_mutually_exclusive_group()
    output_format.add_argument('--mirex',
                               help='use MIREX format for output',
                               action="store_true")
    output_format.add_argument('--jams',
                               help='use JAMS format for output',
                               action="store_true")

    parser.add_argument('-i', '--input',
                        nargs='+',
                        help='input audio file(s) to process')

    output_location = parser.add_mutually_exclusive_group()
    output_location.add_argument('-o', '--output',
                                 nargs='*',
                                 help='output file(s)')
    output_location.add_argument('-d', '--outputdir',
                                 help='output directory')

    # parse arguments
    args = parser.parse_args()

    _check_tempo_args(args, parser)

    # load model
    model = args.model
    print('Loading model \'{}\'...'.format(model))
    classifier = TempoClassifier(model)
    print('Loaded model with {} parameters.'.format(classifier.model.count_params()))

    print('Processing file(s)', end='', flush=True)
    for index, input_file in enumerate(args.input):
        try:
            print('.', end='', flush=True)
            features = read_features(input_file)

            create_jam = args.jams
            create_mirex = args.mirex

            if create_mirex or create_jam:
                t1, t2, s1 = classifier.estimate_mirex(features, interpolate=args.interpolate)
                if create_mirex:
                    result = str(t1) + '\t' + str(t2) + '\t' + str(s1)
                else:
                    result = _create_tempo_jam(input_file, model, s1, t1, t2)
            else:
                tempo = classifier.estimate_tempo(features, interpolate=args.interpolate)
                result = str(tempo)

            _write_tempo_result(result=result,
                                input_file=input_file,
                                output_dir=args.outputdir,
                                output_list=args.output,
                                index=index,
                                append_extension=args.extension,
                                replace_extension=args.replace_extension,
                                create_jam=create_jam)
        except Exception as e:
            if not args.cont:
                print('\nAn error occurred while processing \'{}\':\n{}\n'.format(input_file, e), file=sys.stderr)
                raise e
            else:
                print('E({})'.format(input_file), end='', flush=True)
    print('\nDone')


def _write_tempo_result(result,
                        input_file=None,
                        output_dir=None,
                        output_list=None,
                        index=0,
                        append_extension=None, replace_extension=None,
                        create_jam=False):
    """
    Write the tempo analysis results to a file.

    :param result: results
    :param input_file: input file for these results
    :param output_dir: output directory
    :param output_list: list of output file names
    :param index: index in the ``output_list``
    :param append_extension: file extension to append
    :param replace_extension: file extension to used for replacing an existing extension
    :param create_jam: create JAM or not
    """

    file_dir = dirname(input_file)
    file_name = basename(input_file)
    if output_dir is not None:
        file_dir = output_dir

    # determine output_file name
    output_file = None
    if create_jam:
        base, file_extension = splitext(file_name)
        output_file = join(file_dir, base + '.jams')
    elif append_extension is not None:
        output_file = join(file_dir, file_name + append_extension)
    elif replace_extension is not None:
        base, file_extension = splitext(file_name)
        output_file = join(file_dir, base + replace_extension)
    elif output_list is not None and index < len(output_list):
        output_file = output_list[index]

    # actually writing the output
    if create_jam:
        result.save(output_file)
    elif output_file is None:
        print('\n' + result)
    else:
        with open(output_file, mode='w') as file_name:
            file_name.write(result + '\n')


def _create_tempo_jam(input_file, model, s1, t1, t2):
    result = jams.JAMS()
    y, sr = librosa.load(input_file)
    track_duration = librosa.get_duration(y=y, sr=sr)
    result.file_metadata.duration = track_duration
    result.file_metadata.identifiers = {'file': basename(input_file)}
    tempo_a = jams.Annotation(namespace='tempo', time=0, duration=track_duration)
    tempo_a.annotation_metadata = jams.AnnotationMetadata(
        version=package_version,
        annotation_tools=f'schreiber tempo-cnn (model={model}), '
                         f'https://github.com/hendriks73/tempo-cnn',
        data_source='Hendrik Schreiber, Meinard Müller. A Single-Step Approach to '
                    'Musical Tempo Estimation Using a Convolutional Neural Network. '
                    'In Proceedings of the 19th International Society for Music Information '
                    'Retrieval Conference (ISMIR), Paris, France, Sept. 2018.')
    tempo_a.append(time=0.0,
                   duration=track_duration,
                   value=t1,
                   confidence=s1)
    tempo_a.append(time=0.0,
                   duration=track_duration,
                   value=t2,
                   confidence=(1 - s1))
    result.annotations.append(tempo_a)
    return result


def _check_tempo_args(args, parser):
    if args.output is not None and 0 < len(args.output) != len(args.input):
        print('Number of input files ({}) must match number of output files ({}).\nInput={}\nOutput={}'
              .format(len(args.input), len(args.output), args.input, args.output), file=sys.stderr)
        parser.print_help(file=sys.stderr)
        sys.exit(1)
    if args.input is None:
        print('No input files given.', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        sys.exit(1)


def tempogram():
    """tempogram command line entry point."""

    # define parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The program 'tempogram' estimates local tempi for a given file and displays
    their probability distributions in a graph.
    The underlying algorithm and the model ismir2018 is described in detail in:

    Hendrik Schreiber, Meinard Müller,
    "A single-step approach to musical meter estimation using a
    convolutional neural network"
    Proceedings of the 19th International Society for Music Information
    Retrieval Conference (ISMIR), Paris, France, Sept. 2018.

    Models fcn and cnn are from:

    Hendrik Schreiber,
    "CNN-based tempo estimation"
    Music Information Retrieval Evaluation eXchange (MIREX),
    Paris, France, 2018.

    Model fma2018 is from:

    Hendrik Schreiber,
    "Technical Report: Tempo and Meter Estimation for Greek Folk Music Using
    Convolutional Neural Networks and Transfer Learning"
    8th International Workshop on Folk Music Analysis (FMA),
    Thessaloniki, Greece, June 2018.

    Models deeptemp*, deepsquare* and shallowtemp* are from:

    Hendrik Schreiber, Meinard Müller,
    "Musical Tempo and Key Estimation using Convolutional
    Neural Networks with Directional Filters"
    Sound and Music Computing Conference (SMC),
    Málaga, Spain, 2019.

    Models mazurka, dt_maz_* are from:

    Hendrik Schreiber, Frank Zalkow, Meinard Müller,
    "Modeling and Estimating Local Tempo: A Case Study on Chopin’s Mazurkas"
    Proceedings of the 21st International Society for Music Information
    Retrieval Conference (ISMIR), Montréal, QC, Canada, Oct. 2020.

    License: GNU Affero General Public License v3
    ''')

    parser.add_argument('-v', '--version', action='version', version=f'tempogram {package_version}')
    parser.add_argument('-p', '--png',
                        help='write the tempogram to a file, '
                             'adding the file extension .png to the input file name',
                        action="store_true")
    parser.add_argument('-c', '--csv',
                        help='write the tempogram data to a csv file, '
                             'adding the file extension .csv to the input file name',
                        action="store_true")
    parser.add_argument('-s', '--sharpen',
                        help='sharpen the image to a one-hot representation',
                        action="store_true")
    parser.add_argument('-n', '--norm-frame',
                        help='enable framewise normalization using (max|l1|l2)')
    parser.add_argument('--hop-length',
                        help='hop length between predictions, 1 hop = 0.0464399093s',
                        default=32, type=int)
    parser.add_argument('-m', '--model', nargs='?', default='fcn',
                        help='model name (ismir2018|fma2018|cnn|fcn|mazurka|deeptemp|deepsquare|shallowtemp), defaults to fcn')
    parser.add_argument('audio_file', nargs='+', help='audio file to process')

    # parse arguments
    args = parser.parse_args()

    # load model
    print('Loading model \'{}\'...'.format(args.model))
    classifier = TempoClassifier(args.model)
    print('Loaded model with {} parameters.'.format(classifier.model.count_params()))

    hop_length = args.hop_length
    sr = 11025.0
    fft_hop_length = 512.0
    log_scale = 'fma' in args.model
    min_bpm, max_bpm, max_ylim = _get_tempogram_limits(log_scale)

    print('Processing file(s)', end='', flush=True)
    for file in args.audio_file:
        print('.', end='', flush=True)
        features = read_features(file, hop_length=hop_length, zero_pad=True)
        predictions = classifier.estimate(features)

        norm_frame = args.norm_frame
        if norm_frame is not None:
            predictions = _norm_tempogram_frames(predictions, norm_frame)

        sharpen = args.sharpen
        if sharpen:
            predictions = (predictions.T / np.max(predictions, axis=1)).T
            predictions = np.where(predictions != 1, 0, predictions)

        max_windows = predictions.shape[0] * hop_length
        max_length_in_s = max_windows * (fft_hop_length / sr)
        frame_length = (fft_hop_length / sr) * hop_length

        fig = plt.figure()
        fig.canvas.set_window_title('tempogram: ' + file)
        if args.png:
            fig.set_size_inches(5, 2)

        ax = fig.add_subplot(111)
        ax.set_ylim((0, max_ylim))
        ax.imshow(predictions.T, origin='lower', cmap='Greys', aspect='auto',
                  extent=(0, max_length_in_s, min_bpm, max_bpm))

        if log_scale:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(ScalarFormatter())

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Tempo (BPM)')

        if args.csv:
            _write_tempogram_as_csv(predictions=predictions,
                                    classifier=classifier,
                                    file=file,
                                    frame_length=frame_length,
                                    log_scale=log_scale,
                                    min_bpm=min_bpm,
                                    max_bpm=max_bpm,
                                    sharpen=sharpen)

        if args.png:
            plt.tight_layout()
            fig.savefig(file + '.png', dpi=300)

        else:
            plt.show()

    print('\nDone')


def _norm_tempogram_frames(predictions=None, norm_frame=None):
    norm_order = np.inf
    if 'max' == norm_frame.lower():
        norm_order = np.inf
    elif 'l1' == norm_frame.lower():
        norm_order = 1
    elif 'l2' == norm_frame.lower():
        norm_order = 2
    else:
        print('Unknown norm. Using max norm.', end='', flush=True)
    predictions = (predictions.T / np.linalg.norm(predictions, ord=norm_order, axis=1)).T
    return predictions


def _write_tempogram_as_csv(predictions=None,
                            classifier=None,
                            file=None,
                            frame_length=None,
                            log_scale=False,
                            min_bpm=30,
                            max_bpm=256,
                            sharpen=False):
    csv_file_name = file + '.csv'
    if sharpen:
        # for now simple argmax, we could use quad interpolation instead
        index = np.argmax(predictions, axis=1)
        bpm = classifier.to_bpm(index)
        np.savetxt(csv_file_name,
                   bpm,
                   fmt='%1.2f',
                   delimiter=",",
                   header='Predictions using model \'{}\' '
                          'argmax of tempo distribution '
                          '{}-{} BPM (column) '
                          'log_scale={} '
                          'feature frequency={} Hz '
                          'i.e. {} ms/feature (rows)'
                   .format(classifier.model_name, min_bpm, max_bpm, log_scale,
                           1. / frame_length,
                           frame_length * 1000.))
    else:
        np.savetxt(csv_file_name,
                   predictions,
                   fmt='%1.6f',
                   delimiter=",",
                   header='Predictions using model \'{}\' '
                          '{}-{} BPM (columns) '
                          'log_scale={} '
                          'feature frequency={} Hz '
                          'i.e. {} ms/feature (rows)'
                   .format(classifier.model_name, min_bpm, max_bpm, log_scale,
                           1. / frame_length,
                           frame_length * 1000.))


def _get_tempogram_limits(log_scale):
    if log_scale:
        min_bpm = 50
        max_bpm = 500
        max_ylim = 510
    else:
        min_bpm = 30
        max_bpm = 286
        max_ylim = 300
    return min_bpm, max_bpm, max_ylim


def greekfolk():
    """greekfolk command line entry point."""

    # define parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The program 'greekfolk' estimates the global tempo and the numerator of a
    global meter for Greek folk music tracks.

    The used models are based on an approach described in detail in:

    Hendrik Schreiber, Meinard Müller,
    "A single-step approach to musical meter estimation using a
    convolutional neural network"
    Proceedings of the 19th International Society for Music Information
    Retrieval Conference (ISMIR), Paris, France, Sept. 2018.

    For the purpose of estimating meter and tempo for Greek folk music,
    transfer learning on a small dataset was conducted.
    For details see:

    Hendrik Schreiber,
    "Technical Report: Tempo and Meter Estimation for Greek Folk Music Using
    Convolutional Neural Networks and Transfer Learning"
    8th International Workshop on Folk Music Analysis (FMA),
    Thessaloniki, Greece, June 2018.

    License: GNU Affero General Public License v3
    ''')

    parser.add_argument('-v', '--version', action='version', version=f'greekfolk {package_version}')
    parser.add_argument('input', help='input folder with wav files')
    parser.add_argument('output', help='output folder')

    # parse arguments
    args = parser.parse_args()

    if not exists(args.output):
        print('Creating output dir: ' + args.output)
        makedirs(args.output)

    # load models
    print('Loading models...')
    meter_classifier = MeterClassifier('fma2018-meter')
    tempo_classifier = TempoClassifier('fma2018')

    print('Processing file(s)...')

    wav_files = [join(args.input, f) for f in listdir(args.input) if f.endswith('.wav') and isfile(join(args.input, f))]
    if len(wav_files) == 0:
        print("No .wav files found in " + args.input)
    for input_file in wav_files:
        print('Analyzing: ' + input_file)
        meter_features = read_features(input_file, frames=512, hop_length=256)
        meter_result = str(meter_classifier.estimate_meter(meter_features))

        tempo_features = read_features(input_file)
        tempo_result = str(round(tempo_classifier.estimate_tempo(tempo_features), 1))

        output_file = join(args.output, basename(input_file).replace('.wav', '.txt'))
        with open(output_file, mode='w') as f:
            f.write(tempo_result + '\t' + meter_result + '\n')
    print('\nDone')


def meter():
    """meter command line entry point."""

    # define parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The program 'meter' estimates the numerator of a global meter for a given file.
    The used model is based on an approach described in detail in:

    Hendrik Schreiber, Meinard Müller,
    "A single-step approach to musical meter estimation using a
    convolutional neural network"
    Proceedings of the 19th International Society for Music Information
    Retrieval Conference (ISMIR), Paris, France, Sept. 2018.

    For the purpose of estimating meter and tempo for Greek folk music,
    transfer learning on a small dataset was conducted.
    For details see:

    Hendrik Schreiber,
    "Technical Report: Tempo and Meter Estimation for Greek Folk Music Using
    Convolutional Neural Networks and Transfer Learning"
    8th International Workshop on Folk Music Analysis (FMA),
    Thessaloniki, Greece, June 2018.

    License: GNU Affero General Public License v3
    ''')

    parser.add_argument('-v', '--version', action='version', version=f'meter {package_version}')
    parser.add_argument('-m', '--model', nargs='?', default='fma2018-meter',
                        help='model name [fma2018-meter], defaults to fma2018-meter')
    parser.add_argument('-i', '--input', nargs='+', help='input audio file(s) to process')

    output_options = parser.add_mutually_exclusive_group()
    output_options.add_argument('-o', '--output', nargs='*', help='output file(s)')
    output_options.add_argument('-e', '--extension', help='append given extension to original file name for results')

    # parse arguments
    args = parser.parse_args()

    if args.output is not None and 0 < len(args.output) != len(args.input):
        print('Number of input files must match number of output files.', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        sys.exit(1)

    if args.input is None:
        print('No input files given.', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        sys.exit(1)

    # load model
    print('Loading model...')
    classifier = MeterClassifier(args.model)

    print('Processing file(s)', end='', flush=True)
    for index, input_file in enumerate(args.input):
        print('.', end='', flush=True)
        features = read_features(input_file, frames=512)
        meter = classifier.estimate_meter(features)
        result = str(meter)

        output_file = None
        if args.extension is not None:
            output_file = input_file + args.extension
        elif args.output is not None and index < len(args.output):
            output_file = args.output[index]

        if output_file is None:
            print('\n' + result)
        else:
            with open(output_file, mode='w') as f:
                f.write(result + '\n')
    print('\nDone')
