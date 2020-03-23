"""
Provides helper routines for validating input.
"""

# License: MIT


import numpy as np


def check_unit_axis_sums(a, whom, axis=0):
    """Check sum along array axis is close to one."""

    axis_sums = a.sum(axis=axis)
    if not np.all(np.isclose(axis_sums, 1)):
        raise ValueError(
            'Array with incorrect axis sums passed to %s. '
            'Expected sums along axis %d to be 1.'
            % (whom, axis))


def check_array_shape(a, shape, whom):
    """Check array shape matches given shape."""

    if a.shape != shape:
        raise ValueError(
            'Array with wrong shape passed to %s. '
            'Expected %s, but got %s' % (whom, shape, a.shape))


def check_stochastic_matrix(a, shape, whom, axis=0):
    """Check array is a stochastic matrix with the correct shape."""

    check_array_shape(a, shape, whom)
    check_unit_axis_sums(a, whom, axis=axis)
