.. image:: https://img.shields.io/badge/License-AGPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/agpl-3.0

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1492353.svg
   :target: https://doi.org/10.5281/zenodo.1492353

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3553592.svg
   :target: https://doi.org/10.5281/zenodo.3553592

=========
Tempo-CNN
=========

Tempo-CNN is a simple CNN-based framework for estimating temporal properties
of music tracks.

First and foremost, Tempo-CNN is a tempo estimator. To determine the global tempo of
an audio file, simply run the script

.. code-block:: console

    tempo -i my_audio.wav

You may specify other models and output formats (`mirex <http://www.music-ir.org/mirex/wiki/2018:Audio_Tempo_Estimation>`_,
`JAMS <https://github.com/marl/jams>`_) via command line parameters.

E.g. to create JAMS as output format and the model originally used in the ISMIR 2018
paper [1], please run

.. code-block:: console

    tempo -m ismir2018 --jams -i my_audio.wav

DeepTemp Models
---------------

To use one of the ``DeepTemp`` models from [3] (see also repo
`directional_cnns <https://github.com/hendriks73/directional_cnns>`_), run

.. code-block:: console

    tempo -m deeptemp --jams -i my_audio.wav

or,

.. code-block:: console

    tempo -m deeptemp_k24 --jams -i my_audio.wav

if you want to use a higher capacity model (some ``k``-values are supported).
``deepsquare`` and ``shallowtemp`` models may also be used.

Mazurka Models
--------------

To use DT-Maz models from [4], run

.. code-block:: console

    tempo -m mazurka -i my_audio.wav

This defaults to the model named ``dt_maz_v_fold0``.
You may choose another fold ``[0-4]`` or another split ``[v|m]``.
So to use fold 3 from the *M*-split, use

.. code-block:: console

    tempo -m dt_maz_m_fold3 -i my_audio.wav

Batch Processing
----------------

For batch processing, you may want to run ``tempo`` like this:

.. code-block:: console

    find /your_audio_dir/ -name '*.wav' -print0 | xargs -0 tempo -d /output_dir/ -i

This will recursively search for all ``.wav`` files in ``/your_audio_dir/``, analyze then
and write the results to individual files in ``/output_dir/``. Because the model is only
loaded once, this method of processing is much faster than individual program starts.

Interpolation
-------------

To increase accuracy for greater than integer-precision, you may want to enable quadratic interpolation.
You can do so by setting the ``--interpolate`` flag. Obviously, this only makes sense for tracks
with a very stable tempo:

.. code-block:: console

    tempo -m ismir2018 --interpolate -i my_audio.wav

Tempogram
---------

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

Tempo-CNN provides experimental support for temporal property estimation of Greek
folk music [2]. The corresponding models are named ``fma2018`` (for tempo) and ``fma2018-meter``
(for meter). To estimate the meter's numerator, run

.. code-block:: console

    meter -m fma2018-meter -i my_audio.wav


Installation
============

Clone this repo and run ``setup.py install`` using Python 3.6:

.. code-block:: console

    git clone https://github.com/hendriks73/tempo-cnn.git
    cd tempo-cnn
    python setup.py install

You may need to install TensorFlow using ``pip`` from the command line.

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
      Address = {M{\'a}laga, Spain}
   }

Mazurka models:

.. code-block:: latex

   @inproceedings{SchreiberZM20_LocalTempo_ISMIR,
      Title = {Modeling and Estimating Local Tempo: A Case Study on Chopin’s Mazurkas},
      Author = {Hendrik Schreiber and Frank Zalkow and Meinard M{\"u}ller},
      Booktitle = {Proceedings of the 21th International Society for Music Information Retrieval Conference ({ISMIR})},
      Year = {2020},
      Address = {Montreal, QC, Canada}
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
