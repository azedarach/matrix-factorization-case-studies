Matrix factorization case studies
=================================
[![DOI](https://zenodo.org/badge/249302824.svg)](https://zenodo.org/badge/latestdoi/249302824)

Contains code and notebooks for experiments applying
different matrix factorizations to climate data.

To install from source, run:

    python setup.py install

It is recommended that the package be installed into a custom
environment. For example, to install into a custom conda
environment, first create the environment via

    conda create -n matrix-factorization-env python=3.7
    conda activate matrix-factorization-env

The package may then be installed using

    cd /path/to/package/directory
    python setup.py install

Optionally, a set of unit tests may be run by executing

    python setup.py test
