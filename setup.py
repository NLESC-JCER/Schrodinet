#!/usr/bin/env python

import os

from setuptools import (find_packages, setup)

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit Schrodinet/__version__.py
version = {}
with open(os.path.join(here, 'schrodinet', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='schrodinet',
    version=version['__version__'],
    description="Solving the Schrodinger equation using RBF Neural Net",
    long_description=readme + '\n\n',
    long_description_content_type='text/markdown',
    author=["Nicolas Renaud", "Felipe Zapata"],
    author_email='n.renaud@esciencecenter.nl',
    url='https://github.com/NLESC-JCER/Schrodinet',
    packages=find_packages(),
    package_dir={'schrodinet': 'schrodinet'},
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='schrodinet',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
    test_suite='tests',
    install_requires=['autograd', 'cython', 'matplotlib', 'numpy',
                      'pyyaml>=5.1', 'schema', 'scipy', 'tqdm', 'torch'],
    extras_require={
        'dev': ['prospector[with_pyroma]', 'yapf', 'isort'],
        'doc': ['recommonmark', 'sphinx', 'sphinx_rtd_theme'],
        'test': ['coverage', 'pycodestyle', 'pytest', 'pytest-cov',
                 'pytest-runner'],
    }
)
