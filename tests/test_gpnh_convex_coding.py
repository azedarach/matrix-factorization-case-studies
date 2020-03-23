"""
Provides unit tests for GPNH regularized convex coding.
"""

# License: MIT


import numpy as np

from sklearn.utils import check_random_state

from convex_dim_red import right_stochastic_matrix
from convex_dim_red.gpnh_convex_coding import (
    _gpnh_cost, _iterate_gpnh_convex_coding,
    _update_gpnh_dictionary, _update_gpnh_weights)


def test_cost_returns_zero_for_perfect_reconstruction_no_regularization():
    """Test cost is zero for perfect factorization."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 5
    n_components = 3
    n_samples = 30
    tolerance = 1e-14

    lambda_W = 0

    W = random_state.uniform(size=(n_features, n_components))
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(Z.sum(axis=1), 1, tolerance)

    X = Z.dot(W.T)

    cost = _gpnh_cost(X, Z, W, lambda_W=lambda_W)
    expected_cost = 0

    assert abs(cost - expected_cost) < 1e-14


def test_single_dictionary_update_reduces_cost_function_with_zero_lambda():
    """Test single dictionary update reduces cost function."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 7
    n_components = 5
    n_samples = 450

    lambda_W = 0

    X = random_state.uniform(size=(n_samples, n_features))
    W = random_state.uniform(size=(n_features, n_components))
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(Z.sum(axis=1), 1, 1e-14)

    prefactor = (4.0 / (n_features * n_components * (n_components - 1)))
    GW = prefactor * (n_components * np.eye(n_components) - 1)
    ZtZ = Z.T.dot(Z)

    initial_cost = _gpnh_cost(X, Z, W, lambda_W=lambda_W)

    updated_W = _update_gpnh_dictionary(X, Z, ZtZ, GW, lambda_W=lambda_W)

    final_cost = _gpnh_cost(X, Z, updated_W, lambda_W=lambda_W)

    assert final_cost <= initial_cost


def test_single_dictionary_update_reduces_cost_function_with_nonzero_lambda():
    """Test single dictionary update reduces regularized cost function."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 11
    n_components = 6
    n_samples = 230

    lambda_W = 3.2

    X = random_state.uniform(size=(n_samples, n_features))
    W = random_state.uniform(size=(n_features, n_components))
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(Z.sum(axis=1), 1, 1e-14)

    prefactor = (4.0 / (n_features * n_components * (n_components - 1)))
    GW = prefactor * (n_components * np.eye(n_components) - 1)
    ZtZ = Z.T.dot(Z)

    initial_cost = _gpnh_cost(X, Z, W, lambda_W=lambda_W)

    updated_W = _update_gpnh_dictionary(X, Z, ZtZ, GW, lambda_W=lambda_W)

    final_cost = _gpnh_cost(X, Z, updated_W, lambda_W=lambda_W)

    assert final_cost <= initial_cost


def test_exact_solution_is_dictionary_update_fixed_point():
    """Test exact solution is a fixed point of dictionary update step."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 10
    n_components = 6
    n_samples = 40

    lambda_W = 0

    tolerance = 1e-6

    W = random_state.uniform(size=(n_features, n_components))

    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(Z.sum(axis=1), 1, tolerance)

    X = Z.dot(W.T)

    prefactor = (4.0 / (n_features * n_components * (n_components - 1)))
    GW = prefactor * (n_components * np.eye(n_components) - 1)
    ZtZ = Z.T.dot(Z)

    initial_cost = _gpnh_cost(X, Z, W, lambda_W=lambda_W)

    updated_W = _update_gpnh_dictionary(X, Z, ZtZ, GW, lambda_W=lambda_W)

    final_cost = _gpnh_cost(X, Z, updated_W, lambda_W=lambda_W)

    assert np.allclose(updated_W, W, tolerance)

    assert abs(final_cost - initial_cost) < tolerance


def test_repeated_dictionary_updates_converge_with_zero_lambda():
    """Test repeated updates converge to fixed point."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 13
    n_components = 3
    n_samples = 50

    max_iterations = 100
    tolerance = 1e-6

    lambda_W = 0

    X = random_state.uniform(size=(n_samples, n_features))
    W = random_state.uniform(size=(n_features, n_components))
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(Z.sum(axis=1), 1, 1e-12)

    initial_cost = _gpnh_cost(X, Z, W, lambda_W=lambda_W)

    updated_Z, updated_W, _, n_iter, _, _ = _iterate_gpnh_convex_coding(
        X, Z, W, lambda_W=lambda_W,
        update_weights=False, update_dictionary=True,
        tolerance=tolerance, max_iterations=max_iterations,
        require_monotonic_cost_decrease=True)

    final_cost = _gpnh_cost(X, updated_Z, updated_W, lambda_W=lambda_W)

    assert final_cost <= initial_cost
    assert n_iter < max_iterations


def test_repeated_dictionary_updates_converge_with_nonzero_lambda():
    """Test repeated updates converge to fixed point for regularized problem."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 27
    n_components = 13
    n_samples = 500

    max_iterations = 100
    tolerance = 1e-6

    lambda_W = 1.5

    X = random_state.uniform(size=(n_samples, n_features))
    W = random_state.uniform(size=(n_features, n_components))
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(Z.sum(axis=1), 1, 1e-12)

    initial_cost = _gpnh_cost(X, Z, W, lambda_W=lambda_W)

    updated_Z, updated_W, _, n_iter, _, _ = _iterate_gpnh_convex_coding(
        X, Z, W, lambda_W=lambda_W,
        update_weights=False, update_dictionary=True,
        tolerance=tolerance, max_iterations=max_iterations,
        require_monotonic_cost_decrease=True, verbose=True)

    final_cost = _gpnh_cost(X, updated_Z, updated_W, lambda_W=lambda_W)

    assert final_cost <= initial_cost
    assert n_iter < max_iterations


def test_single_weights_updates_reduces_cost_function_with_zero_lambda():
    """Test single weights update reduces cost function."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 25
    n_components = 6
    n_samples = 300

    lambda_W = 0

    X = random_state.uniform(size=(n_samples, n_features))
    W = random_state.uniform(size=(n_features, n_components))
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(Z.sum(axis=1), 1, 1e-14)

    initial_cost = _gpnh_cost(X, Z, W, lambda_W=lambda_W)

    updated_Z = _update_gpnh_weights(X, Z, W)

    final_cost = _gpnh_cost(X, updated_Z, W, lambda_W=lambda_W)

    assert final_cost <= initial_cost


def test_single_weights_updates_reduces_cost_function_with_nonzero_lambda():
    """Test single weights update reduces cost function."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 43
    n_components = 10
    n_samples = 320

    lambda_W = 4.2

    X = random_state.uniform(size=(n_samples, n_features))
    W = random_state.uniform(size=(n_features, n_components))
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(Z.sum(axis=1), 1, 1e-14)

    initial_cost = _gpnh_cost(X, Z, W, lambda_W=lambda_W)

    updated_Z = _update_gpnh_weights(X, Z, W)

    final_cost = _gpnh_cost(X, updated_Z, W, lambda_W=lambda_W)

    assert final_cost <= initial_cost


def test_exact_solution_is_weights_update_fixed_point_with_zero_lambda():
    """Test exact solution is a fixed point of weights update step."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 14
    n_components = 5
    n_samples = 324

    lambda_W = 0

    tolerance = 1e-6

    W = random_state.uniform(size=(n_features, n_components))

    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(Z.sum(axis=1), 1, tolerance)

    X = Z.dot(W.T)

    initial_cost = _gpnh_cost(X, Z, W, lambda_W=lambda_W)

    updated_Z = _update_gpnh_weights(X, Z, W)

    final_cost = _gpnh_cost(X, updated_Z, W, lambda_W=lambda_W)

    assert np.allclose(Z, updated_Z, tolerance)

    assert abs(final_cost - initial_cost) < tolerance


def test_exact_solution_is_weights_update_fixed_point_with_nonzero_lambda():
    """Test exact solution is a fixed point of regularized weights update step."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 24
    n_components = 8
    n_samples = 200

    lambda_W = 3.8

    tolerance = 1e-6

    W = random_state.uniform(size=(n_features, n_components))

    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(Z.sum(axis=1), 1, tolerance)

    X = Z.dot(W.T)

    initial_cost = _gpnh_cost(X, Z, W, lambda_W=lambda_W)

    updated_Z = _update_gpnh_weights(X, Z, W)

    final_cost = _gpnh_cost(X, updated_Z, W, lambda_W=lambda_W)

    assert np.allclose(Z, updated_Z, tolerance)

    assert abs(final_cost - initial_cost) < tolerance


def test_repeated_weights_updates_converge_with_zero_lambda():
    """Test repeated weights updates converge to fixed point."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 43
    n_components = 3
    n_samples = 100

    max_iterations = 100
    tolerance = 1e-6

    lambda_W = 0

    X = random_state.uniform(size=(n_samples, n_features))
    W = random_state.uniform(size=(n_features, n_components))
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(Z.sum(axis=1), 1, 1e-12)

    initial_cost = _gpnh_cost(X, Z, W, lambda_W=lambda_W)

    updated_Z, updated_W, _, n_iter, _, _ = _iterate_gpnh_convex_coding(
        X, Z, W, lambda_W=lambda_W,
        update_weights=True, update_dictionary=False,
        tolerance=tolerance, max_iterations=max_iterations,
        require_monotonic_cost_decrease=True)

    final_cost = _gpnh_cost(X, updated_Z, updated_W, lambda_W=lambda_W)

    assert final_cost <= initial_cost
    assert n_iter < max_iterations


def test_repeated_weights_updates_converge_with_nonzero_lambda():
    """Test repeated weights updates converge to fixed point for regularized problem."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 12
    n_components = 6
    n_samples = 500

    max_iterations = 100
    tolerance = 1e-6

    lambda_W = 6.2

    X = random_state.uniform(size=(n_samples, n_features))
    W = random_state.uniform(size=(n_features, n_components))
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(Z.sum(axis=1), 1, 1e-12)

    initial_cost = _gpnh_cost(X, Z, W, lambda_W=lambda_W)

    updated_Z, updated_W, _, n_iter, _, _ = _iterate_gpnh_convex_coding(
        X, Z, W, lambda_W=lambda_W,
        update_weights=True, update_dictionary=False,
        tolerance=tolerance, max_iterations=max_iterations,
        require_monotonic_cost_decrease=True)

    final_cost = _gpnh_cost(X, updated_Z, updated_W, lambda_W=lambda_W)

    assert final_cost <= initial_cost
    assert n_iter < max_iterations
