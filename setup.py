from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

_VERSION = '0.`.2'

REQUIRED_PACKAGES = [
    'transformers',
    'datasets',
]

DEPENDENCY_LINKS = [
]

setuptools.setup(
    name='tfds_bert',
    version=_VERSION,
    description='Bert dataset for tensorflow_datasets',
    install_requires=REQUIRED_PACKAGES,
    dependency_links=DEPENDENCY_LINKS,
    url='https://github.com/justhungryman/tfds-bert',
    # license='MIT License',
    package_dir={},
    packages=setuptools.find_packages(exclude=['tests']),
)