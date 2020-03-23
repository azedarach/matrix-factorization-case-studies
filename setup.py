"""Set-up routines for convex coding experiments."""

from setuptools import setup, find_packages


setup(
    name='convex_dim_red',
    version='0.0.1',
    author='Dylan Harries',
    author_email='Dylan.Harries@csiro.au',
    description='Code for convex coding dimension reduction experiments',
    long_description='',
    install_requires=['joblib', 'numba', 'numpy', 'pytest', 'scikit-learn', 'scipy'],
    setup_requires=['pytest-runner', 'pytest-pylint'],
    tests_require=['pytest', 'pytest-cov', 'pylint'],
    packages=find_packages('src'),
    package_dir={'':'src'},
    test_suite='tests',
    zip_safe=False
)
