"""
Provides unit tests for FurthestSum initialization.
"""

# License: MIT


import numpy as np
import pytest

from convex_dim_red import furthest_sum, left_stochastic_matrix


def test_throws_on_nonsquare_dissimilarity_matrix():
    """Test routine raises exception when given non-square input."""

    n_features = 10
    n_samples = 20
    n_components = 2

    X = np.random.uniform(size=(n_samples, n_features))

    with pytest.raises(ValueError):
        furthest_sum(X, n_components, 0)


def test_throws_when_given_out_of_bounds_start_index():
    """Test routine raises exception when given out of bounds index."""

    n_samples = 10
    n_components = 5

    K = np.random.uniform(size=(n_samples, n_samples))

    with pytest.raises(ValueError):
        furthest_sum(K, n_components, n_samples + 10)


def test_throws_when_start_index_is_excluded():
    """Test routine raises exception when start index is excluded."""

    n_samples = 9
    n_components = 8

    K = np.random.uniform(size=(n_samples, n_samples))

    exclude = np.arange(n_samples)

    with pytest.raises(ValueError):
        furthest_sum(K, n_components, 0, exclude)


def test_throws_error_when_not_enough_points():
    """Test routine raises exception when too few points."""

    n_samples = 32
    n_components = 5
    n_exclude = n_samples - n_components + 2

    K = np.random.uniform(size=(n_samples, n_samples))

    exclude = np.arange(n_exclude)

    assert n_components + n_exclude > n_samples

    with pytest.raises(ValueError):
        furthest_sum(K, n_components, n_samples - 1, exclude)


def test_returns_empty_vector_when_no_components_requested():
    """Test routine returns empty result when no points requested."""

    n_samples = 6
    n_components = 0

    K = np.random.uniform(size=(n_samples, n_samples))

    result = furthest_sum(K, n_components, 0)

    assert len(result) == 0


def test_returns_all_indices_when_number_of_components_equals_number_of_points():
    """Test routine returns all indices when all points requested."""

    n_samples = 20
    n_components = n_samples

    K = np.random.uniform(size=(n_samples, n_samples))

    result = furthest_sum(K, n_components, 5)
    result = sorted(result)

    has_duplicates = False
    found_indices = []
    for index in result:
        for found_index in found_indices:
            if index == found_index:
                has_duplicates = True
                break
        found_indices.append(index)

    assert not has_duplicates

    expected = np.arange(n_components)

    for i in range(n_components):
        assert expected[i] == result[i]


def test_returns_correct_index_when_only_one_sample_present():
    """Test routine returns only index when sample size is one."""

    n_samples = 1
    n_components = 1

    K = np.random.uniform(size=(n_samples, n_samples))

    result = furthest_sum(K, n_components, 0)

    assert len(result) == 1
    assert result[0] == 0


def test_returns_non_excluded_index_when_only_possible():
    """Test routine returns only non-excluded index."""

    n_samples = 102
    n_components = 1

    K = np.random.uniform(size=(n_samples, n_samples))

    leave_in_index = 74
    exclude = [i for i in range(n_samples) if i != leave_in_index]

    result = furthest_sum(K, n_components, leave_in_index, exclude)

    assert len(result) == 1
    assert result[0] == leave_in_index


def test_selects_correct_elements_out_of_three_for_all_starting_points():
    """Test routine selects correct points in small test case."""

    n_samples = 3
    n_components = 2
    max_extra_steps = 10
    exclude = []

    K = np.array([[0, 1, 2], [1, 0, 0.5], [2, 0.5, 0]])

    expected = np.array([0, 2])
    for i in range(n_samples):
        for x in range(1, max_extra_steps + 1):
            result = furthest_sum(K, n_components, i, exclude, x)
            result = sorted(result)

            assert len(result) == n_components

            for j in range(n_components):
                assert result[j] == expected[j]


def test_selects_elements_in_convex_hull():
    """Test routine correctly selects points in convex hull."""

    n_samples = 10

    basis = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                     dtype='f8')
    n_basis = basis.shape[0]  # pylint: disable=unsubscriptable-object

    weights = left_stochastic_matrix((n_samples, n_basis))

    assignments = [0, 4, 6, 9]
    for i in range(n_basis):
        weights[assignments[i]] = np.zeros(n_basis)
        weights[assignments[i], i] = 1

    X = weights.dot(basis)
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.linalg.norm(X[i] - X[j])

    n_components = basis.shape[0]   # pylint: disable=unsubscriptable-object

    result = furthest_sum(K, n_components, 1)
    result = sorted(result)

    assert len(result) == n_components
    for i in range(n_components):
        assert result[i] == assignments[i]
