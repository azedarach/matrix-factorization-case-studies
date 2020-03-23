"""
Provides unit test for SPG optimization routine.
"""

# License: MIT


import numpy as np

from convex_dim_red import spg


def test_correct_solution_on_unconstrained_1d_trivial_problem():
    """Test finds correct solution for trivial 1D minimization problem."""

    tolerance = 1e-10
    max_iterations = 100
    max_feval = 100

    def f(x):
        return x * x

    def df(x):
        return 2.0 * x

    x0 = np.random.uniform(-10.0, 10.0)

    x, f_min, n_iter, n_feval = spg(
        f, df, x0, max_iterations=max_iterations, max_feval=max_feval)

    assert abs(x) < tolerance
    assert abs(f_min) < tolerance
    assert n_iter < max_iterations
    assert n_feval < max_feval


def test_correct_solution_on_constrained_1d_trivial_problem():
    """Test finds correct solution for trivial constrained 1D problem.

    The test problem is a quartic with a local minimum at x = 0 and
    a global minimum at x = 2, with the feasible region defined as
    the interval [-1, 0.5].
    """

    tolerance = 1e-6
    max_iterations = 100
    max_feval = 100

    a = 1.0
    b = -15.0 / 4.0
    c = 13.0 / 4.0
    d = 0.0
    e = 1.0

    def f(x):
        return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

    def df(x):
        return 4 * a * x ** 3 + 3 * b * x ** 2 + 2 * c * x + d

    def project(x):
        if x < -1:
            return -1

        if x > 0.5:
            return 0.5

        return x

    x0 = np.random.uniform(1.1, 3.0)

    x, f_min, n_iter, n_feval = spg(
        f, df, x0, project=project,
        max_iterations=max_iterations, max_feval=max_feval)

    assert abs(x) < tolerance
    assert abs(f_min - 1) < tolerance
    assert n_iter < max_iterations
    assert n_feval < max_feval

    x0 = np.random.uniform(-5.0, -2.0)

    x, f_min, n_iter, n_feval = spg(
        f, df, x0, project=project,
        max_iterations=max_iterations, max_feval=max_feval)

    assert abs(x) < tolerance
    assert abs(f_min - 1) < tolerance
    assert n_iter < max_iterations
    assert n_feval < max_feval
