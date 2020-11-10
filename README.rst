.. image:: https://img.shields.io/badge/License-AGPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/agpl-3.0

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1492353.svg
   :target: https://doi.org/10.5281/zenodo.1492353

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3553592.svg
   :target: https://doi.org/10.5281/zenodo.3553592

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3249250.svg
   :target: https://doi.org/10.5281/zenodo.3249250

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4245546.svg
   :target: https://doi.org/10.5281/zenodo.4245546

.. image:: https://github.com/hendriks73/tempo-cnn/workflows/Build%20and%20Test/badge.svg
   :target: https://github.com/hendriks73/tempo-cnn/actions

.. image:: https://badge.fury.io/py/tempocnn.svg
    :target: https://badge.fury.io/py/tempocnn

=========
Tempo-CNN
=========

Tempo-CNN is a simple CNN-based framework for estimating temporal properties
of music tracks featuring trained models from several publications
[1]_ [2]_ [3]_ [4]_.

First and foremost, Tempo-CNN is a tempo estimator. To determine the *global* tempo of
an audio file, simply run the script

.. code-block:: console

    tempo -i my_audio.wav

To create a *local* tempo `"tempogram" <#tempogram>`_, run

.. code-block:: console

    tempogram my_audio.wav

For a complete list of options, run either script with the parameter ``--help``.

For programmatic use via the Python API, please see `here <#programmatic-usage>`_.

Installation
============

In a clean Python 3.6 or 3.7 environment, simply run:

.. code-block:: console

    pip install tempocnn


If you rather want to install from source, clone this repo and run
``setup.py install`` using Python 3.6 or 3.7:

.. code-block:: console

    git clone https://github.com/hendriks73/tempo-cnn.git
    cd tempo-cnn
    python setup.py install


Models and Formats
==================

You may specify other models and output formats (`MIREX <http://www.music-ir.org/mirex/wiki/2018:Audio_Tempo_Estimation>`_,
`JAMS <https://github.com/marl/jams>`_) via command line parameters.

E.g. to create JAMS as output format and the model originally used in the ISMIR 2018
paper [1]_, please run

.. code-block:: console

    tempo -m ismir2018 --jams -i my_audio.wav

For MIREX-style output, add the ``--mirex`` parameter.


DeepTemp Models
===============

To use one of the ``DeepTemp`` models from [3]_ (see also repo
`directional_cnns <https://github.com/hendriks73/directional_cnns>`_), run

.. code-block:: console

    tempo -m deeptemp --jams -i my_audio.wav

or,

.. code-block:: console

    tempo -m deeptemp_k24 --jams -i my_audio.wav

if you want to use a higher capacity model (some ``k``-values are supported).
``deepsquare`` and ``shallowtemp`` models may also be used.

Note that some models may be downloaded (and cached) at execution time.

Mazurka Models
==============

To use DT-Maz models from [4]_, run

.. code-block:: console

    tempo -m mazurka -i my_audio.wav

This defaults to the model named ``dt_maz_v_fold0``.
You may choose another fold ``[0-4]`` or another split ``[v|m]``.
So to use fold 3 from the *M*-split, use

.. code-block:: console

    tempo -m dt_maz_m_fold3 -i my_audio.wav

Note that Mazurka models may be used to estimate a global tempo, but were
actually trained to create `tempograms <#tempogram>`_ for Chopin
Mazurkas [4]_.

While it's cumbersome to list the split definitions for the Version folds,
the Mazurka folds are easily defined:

- ``fold0`` was tested on ``Chopin_Op068No3`` and validated on ``Chopin_Op017No4``
- ``fold1`` was tested on ``Chopin_Op017No4`` and validated on ``Chopin_Op024No2``
- ``fold2`` was tested on ``Chopin_Op024No2`` and validated on ``Chopin_Op030No2``
- ``fold3`` was tested on ``Chopin_Op030No2`` and validated on ``Chopin_Op063No3``
- ``fold4`` was tested on ``Chopin_Op063No3`` and validated on ``Chopin_Op068No3``

The networks were trained on recordings of the three remaining Mazurkas.
In essence this means, **do not** estimate the local tempo for ``Chopin_Op024No2`` using
``dt_maz_m_fold0``, because ``Chopin_Op024No2`` was used in training.

Batch Processing
================

For batch processing, you may want to run ``tempo`` like this:

.. code-block:: console

    find /your_audio_dir/ -name '*.wav' -print0 | xargs -0 tempo -d /output_dir/ -i

This will recursively search for all ``.wav`` files in ``/your_audio_dir/``, analyze then
and write the results to individual files in ``/output_dir/``. Because the model is only
loaded once, this method of processing is much faster than individual program starts.

Interpolation
=============

To increase accuracy for greater than integer-precision, you may want to enable quadratic interpolation.
You can do so by setting the ``--interpolate`` flag. Obviously, this only makes sense for tracks
with a very stable tempo:

.. code-block:: console

    tempo -m ismir2018 --interpolate -i my_audio.wav

Tempogram
=========

Instead of estimating a global tempo, Tempo-CNN can also estimate local tempi in the
form of a tempogram. This can be useful for identifying tempo drift.

To create such a tempogram, run

.. code-block:: console

    tempogram -p my_audio.wav

As output, ``tempogram`` will create a ``.png`` file. Additional options to select different models
and output formats are available.

You may use the ``--csv`` option to export local tempo estimates in a parseable format and the
``--hop-length`` option to change temporal resolution.
The parameters ``--sharpen`` and ``--norm-frame`` let you post-process the image.


Greek Folk
==========

Tempo-CNN provides experimental support for temporal property estimation of Greek
folk music [2]_. The corresponding models are named ``fma2018`` (for tempo) and ``fma2018-meter``
(for meter). To estimate the meter's numerator, run

.. code-block:: console

    meter -m fma2018-meter -i my_audio.wav

Programmatic Usage
==================

After `installation <#installation>`_, you may use
the package programmatically.

Example for *global* tempo estimation:

.. code-block:: python

    from tempocnn.classifier import TempoClassifier
    from tempocnn.feature import read_features

    model_name = 'cnn'
    input_file = 'some_audio_file.mp3'

    # initialize the model (may be re-used for multiple files)
    classifier = TempoClassifier(model_name)

    # read the file's features
    features = read_features(input_file)

    # estimate the global tempo
    tempo = classifier.estimate_tempo(features, interpolate=False)
    print(f"Estimated global tempo: {tempo}")


Example for *local* tempo estimation:


.. code-block:: python

    from tempocnn.classifier import TempoClassifier
    from tempocnn.feature import read_features

    model_name = 'cnn'
    input_file = 'some_audio_file.mp3'

    # initialize the model (may be re-used for multiple files)
    classifier = TempoClassifier(model_name)

    # read the file's features, specify hop_length for temporal resolution
    features = read_features(input_file, frames=256, hop_length=32)

    # estimate local tempi, this returns tempo classes, i.e., a distribution
    local_tempo_classes = classifier.estimate(features)

    # find argmax per frame and convert class index to BPM value
    max_predictions = np.argmax(local_tempo_classes, axis=1)
    local_tempi = classifier.to_bpm(max_predictions)
    print(f"Estimated local tempo classes: {local_tempi}")


License
=======

Source code and models can be licensed under the GNU AFFERO GENERAL PUBLIC LICENSE v3.
For details, please see the `LICENSE <LICENSE>`_ file.


Citation
========

If you use Tempo-CNN in your work, please consider citing it.

Original publication:

.. code-block:: latex

   @inproceedings{SchreiberM18_TempoCNN_ISMIR,
      Title = {A Single-Step Approach to Musical Tempo Estimation Using a Convolutional Neural Network},
      Author = {Schreiber, Hendrik and M{\"u}ller Meinard},
      Booktitle = {Proceedings of the 19th International Society for Music Information Retrieval Conference ({ISMIR})},
      Pages = {98--105},
      Month = {9},
      Year = {2018},
      Address = {Paris, France},
      doi = {10.5281/zenodo.1492353},
      url = {https://doi.org/10.5281/zenodo.1492353}
   }

ShallowTemp, DeepTemp, and DeepSquare models:

.. code-block:: latex

   @inproceedings{SchreiberM19_CNNKeyTempo_SMC,
      Title = {Musical Tempo and Key Estimation using Convolutional Neural Networks with Directional Filters},
      Author = {Hendrik Schreiber and Meinard M{\"u}ller},
      Booktitle = {Proceedings of the Sound and Music Computing Conference ({SMC})},
      Pages = {47--54},
      Year = {2019},
      Address = {M{\'a}laga, Spain},
      doi = {10.5281/zenodo.3249250},
      url = {https://doi.org/10.5281/zenodo.3249250}
   }

Mazurka models:

.. code-block:: latex

   @inproceedings{SchreiberZM20_LocalTempo_ISMIR,
      Title = {Modeling and Estimating Local Tempo: A Case Study on Chopin’s Mazurkas},
      Author = {Hendrik Schreiber and Frank Zalkow and Meinard M{\"u}ller},
      Booktitle = {Proceedings of the 21th International Society for Music Information Retrieval Conference ({ISMIR})},
      Pages = {773--779},
      Year = {2020},
      Address = {Montreal, QC, Canada},
      doi = {10.5281/zenodo.4245546},
      url = {https://doi.org/10.5281/zenodo.4245546}
   }

References
==========

.. [1] Hendrik Schreiber, Meinard Müller, `A Single-Step Approach to Musical Tempo Estimation
    Using a Convolutional Neural Network <https://zenodo.org/record/1492353/files/141_Paper.pdf>`_,
    Proceedings of the 19th International Society for Music Information
    Retrieval Conference (ISMIR), Paris, France, Sept. 2018.
.. [2] Hendrik Schreiber, `Technical Report: Tempo and Meter Estimation for
    Greek Folk Music Using Convolutional Neural Networks and Transfer Learning
    <http://www.tagtraum.com/download/2018_SchreiberGreekFolkTempoMeter.pdf>`_,
    8th International Workshop on Folk Music Analysis (FMA),
    Thessaloniki, Greece, June 2018.
.. [3] Hendrik Schreiber, Meinard Müller, `Musical Tempo and Key Estimation using Convolutional
    Neural Networks with Directional Filters
    <http://smc2019.uma.es/articles/P1/P1_07_SMC2019_paper.pdf>`_,
    Proceedings of the Sound and Music Computing Conference (SMC),
    Málaga, Spain, 2019.
.. [4] Hendrik Schreiber, Frank Zalkow, Meinard Müller,
    `Modeling and Estimating Local Tempo: A Case Study on Chopin’s
    Mazurkas <https://program.ismir2020.net/static/final_papers/14.pdf>`_,
    Proceedings of the 21st International Society for Music Information
    Retrieval Conference (ISMIR), Montréal, QC, Canada, Oct. 2020.
