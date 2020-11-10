=======
Changes
=======

0.0.7:
 - Added DOIs to bibtex entries.
 - Added README DOI badge for SMC paper.

0.0.6:
 - Require h5py<3.0.0, to avoid model loading issues.

0.0.5:
 - Moved to TensorFlow 1.15.4.
 - Consolidated version info.
 - Consolidated requirements.
 - Switched to pytest.
 - Officially support Python 3.7.
 - Enabled GitHub actions for packaging and testing.
 - Added Pypi workflow.
 - Cache models locally.
 - Load models from GitHub.
 - Turned off TensorFlow debug logging.
 - Migrated scripts to entry points.
 - Removed charset encoding comments.

0.0.4:
 - Added support for DeepTemp, DeepSquare, and ShallowTemp models.
 - Added support for Mazurka models.
 - Added support for exporting data from tempograms.
 - Added support for framewise normalization in tempograms.
 - Moved to TensorFlow 1.15.2.
 - Print number of model parameters.

0.0.3:
 - Added flag ``--interpolate`` for ``tempo`` to increase accuracy.
 - Migrated models to TensorFlow 1.10.1.

0.0.2:
 - Added ``-d`` option for improved batch processing (tempo)
 - Improved jams output
 - Moved to librosa 0.6.2
 - Continue processing batch, even when encountering an error

0.0.1:
 - Initial version