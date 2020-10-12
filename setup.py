#!/usr/bin/env python
# encoding: utf-8
"""
This file contains the setup for setuptools to distribute everything as a
(PyPI) package.

"""

from setuptools import setup, find_packages
from importlib.machinery import SourceFileLoader

import glob

# define version
version = SourceFileLoader("tempocnn.version", "tempocnn/version.py").load_module()
version = version.__version__

# define scripts to be installed by the PyPI package
scripts = glob.glob('bin/*')

# define the models to be included in the PyPI package
# do not package some large models, to stay below PyPI 100mb threshold
package_data = [
    'models/cnn.h5',
    'models/fcn.h5',
    'models/ismir2018.h5',
#    'models/fma2018.h5',
#    'models/fma2018-meter.h5',
    'models/dt_maz_m_fold0.h5',
    'models/dt_maz_m_fold1.h5',
    'models/dt_maz_m_fold2.h5',
    'models/dt_maz_m_fold3.h5',
    'models/dt_maz_m_fold4.h5',
    'models/dt_maz_v_fold0.h5',
    'models/dt_maz_v_fold1.h5',
    'models/dt_maz_v_fold2.h5',
    'models/dt_maz_v_fold3.h5',
    'models/dt_maz_v_fold4.h5',
    'models/deepsquare_k1.h5',
    'models/deepsquare_k2.h5',
    'models/deepsquare_k4.h5',
    'models/deepsquare_k8.h5',
#    'models/deepsquare_k16.h5',
#    'models/deepsquare_k24.h5',
    'models/deeptemp_k2.h5',
    'models/deeptemp_k4.h5',
    'models/deeptemp_k8.h5',
#    'models/deeptemp_k16.h5',
#    'models/deeptemp_k24.h5',
    'models/shallowtemp_k1.h5',
    'models/shallowtemp_k2.h5',
    'models/shallowtemp_k4.h5',
    'models/shallowtemp_k6.h5',
#    'models/shallowtemp_k8.h5',
#    'models/shallowtemp_k12.h5',
]

# requirements
with open('requirements.txt', 'r') as fh:
    requirements = fh.read().splitlines()

# docs to be included
try:
    long_description = open('README.rst', encoding='utf-8').read()
    long_description += '\n' + open('CHANGES.rst', encoding='utf-8').read()
except TypeError:
    long_description = open('README.rst').read()
    long_description += '\n' + open('CHANGES.rst').read()

# the actual setup routine
setup(name='tempocnn',
      version=version,
      description='Python audio signal processing library for musical tempo detection',
      long_description=long_description,
      author='Hendrik Schreiber '
             'tagtraum industries incorporated, '
             'Raleigh, NC, USA',
      author_email='hs@tagtraum.com',
      url='https://github.com/hendriks73/tempo-cnn',
      license='AGPL',
      packages=find_packages(exclude=['test', 'docs']),
      package_data={'tempocnn': package_data},
      exclude_package_data={'': ['tests', 'docs']},
      scripts=scripts,
      python_requires='>=3.6',
      install_requires=requirements,
      extras_require={
          "testing": [
              "pytest",
              "coverage",
          ]
      },
      classifiers=['Development Status :: 3 - Alpha',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Environment :: Console',
                   'License :: OSI Approved :: GNU Affero General Public License v3',
                   'Topic :: Multimedia :: Sound/Audio :: Analysis',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
