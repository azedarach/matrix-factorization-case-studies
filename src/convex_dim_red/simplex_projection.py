"""
Provides routines for projections onto probability simplices.
"""

# License: MIT


import numpy as np

from numba import guvectorize, jit, float64


@jit(nopython=True)
def simplex_project_vector(x):
    """Project vector onto standard simplex."""

    sorted_x = np.sort(x)

    n = sorted_x.size

    t_hat = 0
    for i in range(n - 2, -2, -1):
        t_hat = (sorted_x[-(n - 1 - i):].sum() - 1) / (n - 1 - i)
        if t_hat >= sorted_x[i]:
            break

    return np.fmax(x - t_hat, 0)


@guvectorize([(float64[:, :], float64[:, :])], '(m, n) -> (m, n)',
             target='parallel', nopython=True)
def simplex_project_columns(A, projected):
    """Project columns of matrix onto standard simplex."""

    n_cols = A.shape[1]
    for i in range(n_cols):
        projected[:, i] = simplex_project_vector(A[:, i])


@guvectorize([(float64[:, :], float64[:, :])], '(m, n) -> (m, n)',
             target='parallel', nopython=True)
def simplex_project_rows(A, projected):
    """Project rows of matrix onto standard simplex."""

    n_rows = A.shape[0]
    for i in range(n_rows):
        projected[i] = simplex_project_vector(A[i])
