"""
Provides unit tests for simplex projections.
"""

# License: MIT


import numpy as np

from convex_dim_red.simplex_projection import (simplex_project_rows,
                                               simplex_project_vector)

def test_correct_projection_for_1D_vector():
    """Test single value correctly projected."""

    x = np.array([-0.5])

    expected = np.array([1.])

    x = simplex_project_vector(x)

    assert np.all(x == expected)


def test_1D_vector_in_simplex_invariant():
    """Test does not change vector already in simplex."""

    x = np.array([1.0])

    projection = simplex_project_vector(x)

    assert np.all(x == projection)


def test_returns_correct_projection_for_2D_vector():
    """Test 2D vector is correctly projected."""

    x = np.array([0.8, 0.8])
    expected = np.array([0.5, 0.5])

    x = simplex_project_vector(x)

    assert np.all(x == expected)

    x = np.array([0.0, 2.0])
    expected = np.array([0.0, 1.0])

    x = simplex_project_vector(x)

    assert np.all(x == expected)

    x = np.array([0.5, -0.5])
    expected = np.array([1.0, 0.0])

    x = simplex_project_vector(x)

    assert np.all(x == expected)


def test_2D_vector_in_simplex_invariant():
    """Test does not change 2D vector already in simplex."""

    x = np.array([0.4, 0.6])

    projection = simplex_project_vector(x)

    assert np.all(x == projection)


def test_5D_vector_in_simplex():
    """Test 5D vector projected into simplex."""

    n_features = 5
    tolerance = 1e-14

    x = np.random.uniform(size=(n_features,))
    x = simplex_project_vector(x)

    assert np.all(x >= 0)

    s = x.sum()

    assert np.abs(s - 1) < tolerance


def test_10D_vector_in_simplex():
    """Test 10D vector projected into simplex."""

    n_features = 10
    tolerance = 1e-14

    x = np.random.uniform(size=(n_features,))
    x = simplex_project_vector(x)

    assert np.all(x >= 0)

    s = x.sum()

    assert np.abs(s - 1) < tolerance


def test_100D_vector_in_simplex():
    """Test 100D vector projected into simplex."""

    n_features = 100
    tolerance = 1e-14

    x = np.random.uniform(size=(n_features,))
    x = simplex_project_vector(x)

    assert np.all(x >= 0)

    s = x.sum()

    assert np.abs(s - 1) < tolerance


def test_1D_rows_in_simplex_invariant():
    """Test 1D rows in simplex unchanged."""

    n_features = 1
    n_samples = 15

    X = np.ones((n_samples, n_features))

    projection = simplex_project_rows(X) # pylint: disable=no-value-for-parameter

    assert np.all(projection == X)


def test_correctly_projects_1D_rows():
    """Test 1D rows correctly projected."""

    n_features = 1
    n_samples = 50
    tolerance = 1e-15

    X = np.random.uniform(size=(n_samples, n_features))
    X = simplex_project_rows(X) # pylint: disable=no-value-for-parameter

    expected = np.ones((n_samples, n_features))

    assert np.allclose(X, expected, tolerance)


def test_2D_rows_in_simplex_invariant():
    """Test 2D rows in simplex unchanged."""

    n_features = 2
    n_samples = 10
    tolerance = 1e-15

    X = np.random.uniform(size=(n_samples, n_features))

    row_sums = X.sum(axis=1)
    X = X / row_sums[:, np.newaxis]

    row_sums = X.sum(axis=1)
    assert np.allclose(row_sums, 1)

    projection = simplex_project_rows(X) # pylint: disable=no-value-for-parameter

    assert np.allclose(X, projection, tolerance)


def test_correctly_projects_2D_rows():
    """Test 2D rows correctly projected."""

    tolerance = 1e-15

    X = np.array([[0.5, 0.5], [0.5, 1.0], [0.0, -0.5]])
    expected = np.array([[0.5, 0.5], [0.25, 0.75], [0.75, 0.25]])

    X = simplex_project_rows(X) # pylint: disable=no-value-for-parameter

    assert np.allclose(X, expected, tolerance)


def test_5D_projection_in_simplex():
    """Test 5D rows are projected into simplex."""

    n_features = 5
    n_samples = 57
    tolerance = 1e-15

    X = np.random.uniform(size=(n_samples, n_features))
    X = simplex_project_rows(X) # pylint: disable=no-value-for-parameter

    assert np.all(X >= 0)

    row_sums = X.sum(axis=1)
    assert np.allclose(row_sums, 1, tolerance)


def test_317D_projection_in_simplex():
    """Test 317D rows are projected into simplex."""

    n_features = 317
    n_samples = 341
    tolerance = 1e-14

    X = np.random.uniform(size=(n_samples, n_features))
    X = simplex_project_rows(X) # pylint: disable=no-value-for-parameter

    assert np.all(X >= 0)

    row_sums = X.sum(axis=1)
    assert np.allclose(row_sums, 1, tolerance)
