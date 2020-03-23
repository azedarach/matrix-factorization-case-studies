"""
Provides unit tests for archetypal analysis routines.
"""

# License: MIT


from __future__ import absolute_import, division, print_function

import numpy as np

from sklearn.utils import check_random_state

from convex_dim_red.archetypal_analysis import (
    _iterate_kernel_aa,
    _kernel_aa_cost,
    _update_kernel_aa_dictionary,
    _update_kernel_aa_weights)
from convex_dim_red import KernelAA, right_stochastic_matrix


def test_single_dictionary_update_reduces_cost_with_zero_delta():
    """Test single update step reduces cost function."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 10
    n_components = 5
    n_samples = 400

    X = random_state.uniform(size=(n_samples, n_features))
    K = X.dot(X.T)

    C = right_stochastic_matrix(
        (n_components, n_samples), random_state=random_state)
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)
    alpha = np.ones(n_components)

    trace_K = np.trace(K)
    KZ = K.dot(Z)
    ZtZ = Z.T.dot(Z)

    assert np.allclose(C.sum(axis=1), 1, 1e-12)
    assert np.allclose(Z.sum(axis=1), 1, 1e-12)

    initial_cost = _kernel_aa_cost(K, Z, C, alpha)

    updated_C = _update_kernel_aa_dictionary(
        K, C, alpha, trace_K, KZ, ZtZ)

    final_cost = _kernel_aa_cost(K, Z, updated_C, alpha)

    assert final_cost <= initial_cost
    assert np.allclose(updated_C.sum(axis=1), 1, 1e-12)


def test_single_dictionary_update_reduces_cost_with_nonzero_delta():
    """Test single update step reduces cost function."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 10
    n_components = 5
    n_samples = 400
    delta = 0.1

    X = random_state.uniform(size=(n_samples, n_features))
    K = X.dot(X.T)

    C = right_stochastic_matrix(
        (n_components, n_samples), random_state=random_state)
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)
    alpha = np.random.uniform(low=(1 - delta), high=(1 + delta),
                              size=(n_components,))

    trace_K = np.trace(K)
    KZ = K.dot(Z)
    ZtZ = Z.T.dot(Z)

    assert np.allclose(C.sum(axis=1), 1, 1e-12)
    assert np.allclose(Z.sum(axis=1), 1, 1e-12)

    initial_cost = _kernel_aa_cost(K, Z, C, alpha)

    updated_C = _update_kernel_aa_dictionary(
        K, C, alpha, trace_K, KZ, ZtZ)

    final_cost = _kernel_aa_cost(K, Z, updated_C, alpha)

    assert final_cost <= initial_cost
    assert np.allclose(updated_C.sum(axis=1), 1, 1e-12)

def test_exact_solution_with_zero_delta_is_dictionary_update_fixed_point():
    """Test exact solution is a fixed point of the dictionary update."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 10
    n_components = 6
    n_samples = 100
    tolerance = 1e-12

    basis = random_state.uniform(size=(n_components, n_features))

    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    archetype_indices = np.zeros(n_components, dtype='i8')
    for i in range(n_components):
        new_index = False
        current_index = 0

        while not new_index:
            new_index = True

            current_index = random_state.randint(
                low=0, high=n_samples)

            for index in archetype_indices:
                if current_index == index:
                    new_index = False

        archetype_indices[i] = current_index

    C = np.zeros((n_components, n_samples))
    component = 0
    for index in archetype_indices:
        C[component, index] = 1.0
        for i in range(n_components):
            if i == component:
                Z[index, i] = 1.0
            else:
                Z[index, i] = 0.0
        component += 1

    X = Z.dot(basis)
    basis_projection = C.dot(X)

    assert np.allclose(basis_projection, basis, tolerance)
    assert np.linalg.norm(X - Z.dot(C.dot(X))) < tolerance

    K = X.dot(X.T)

    alpha = np.ones(n_components)

    initial_cost = _kernel_aa_cost(K, Z, C, alpha)

    trace_K = np.trace(K)
    KZ = K.dot(Z)
    ZtZ = Z.T.dot(Z)

    updated_C = _update_kernel_aa_dictionary(
        K, C, alpha, trace_K, KZ, ZtZ)

    final_cost = _kernel_aa_cost(K, Z, updated_C, alpha)

    assert abs(final_cost - initial_cost) < tolerance

    assert np.allclose(updated_C.sum(axis=1), 1, 1e-12)
    assert np.allclose(updated_C, C, tolerance)


def test_repeated_dictionary_updates_converge_with_zero_delta():
    """Test repeated updates converge to a fixed point with delta = 0."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 20
    n_components = 15
    n_samples = 600
    max_iterations = 1000
    tolerance = 1e-6

    X = random_state.uniform(size=(n_samples, n_features))
    K = X.dot(X.T)

    C = right_stochastic_matrix(
        (n_components, n_samples), random_state=random_state)
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(C.sum(axis=1), 1, tolerance)
    assert np.allclose(Z.sum(axis=1), 1, tolerance)

    delta = 0
    alpha = np.ones(n_components)

    initial_cost = _kernel_aa_cost(K, Z, C, alpha)

    updated_Z, updated_C, updated_alpha, _, n_iter = _iterate_kernel_aa(
        K, Z, C, alpha, delta=delta,
        update_weights=False, update_dictionary=True,
        update_scale_factors=False,
        tolerance=tolerance, max_iterations=max_iterations,
        require_monotonic_cost_decrease=True)[:5]

    final_cost = _kernel_aa_cost(K, updated_Z, updated_C, updated_alpha)

    assert final_cost <= initial_cost
    assert n_iter < max_iterations

    assert np.allclose(updated_Z, Z, 1e-12)
    assert np.allclose(updated_alpha, alpha, 1e-12)
    assert np.allclose(updated_C.sum(axis=1), 1, 1e-12)


def test_repeated_dictionary_updates_converge_with_nonzero_delta():
    """Test repeated updates converge to a fixed point with non-zero delta."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 20
    n_components = 15
    n_samples = 600
    max_iterations = 1000
    tolerance = 1e-6

    X = random_state.uniform(size=(n_samples, n_features))
    K = X.dot(X.T)

    C = right_stochastic_matrix(
        (n_components, n_samples), random_state=random_state)
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(C.sum(axis=1), 1, tolerance)
    assert np.allclose(Z.sum(axis=1), 1, tolerance)

    delta = 0.2
    alpha = random_state.uniform(low=(1 - delta), high=(1 + delta),
                                 size=(n_components,))

    initial_cost = _kernel_aa_cost(K, Z, C, alpha)

    updated_Z, updated_C, updated_alpha, _, n_iter = _iterate_kernel_aa(
        K, Z, C, alpha, delta=delta,
        update_weights=False, update_dictionary=True,
        update_scale_factors=False,
        tolerance=tolerance, max_iterations=max_iterations,
        require_monotonic_cost_decrease=True)[:5]

    final_cost = _kernel_aa_cost(K, updated_Z, updated_C, updated_alpha)

    assert final_cost <= initial_cost
    assert n_iter < max_iterations

    assert np.allclose(updated_Z, Z, 1e-12)
    assert np.allclose(updated_alpha, alpha, 1e-12)
    assert np.allclose(updated_C.sum(axis=1), 1, 1e-12)


def test_single_weights_update_reduces_cost_with_zero_delta():
    """Test single weights update reduces cost function with delta = 0."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 13
    n_components = 7
    n_samples = 100

    X = random_state.uniform(size=(n_samples, n_features))
    K = X.dot(X.T)

    C = right_stochastic_matrix(
        (n_components, n_samples), random_state=random_state)
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)
    alpha = np.ones(n_components)

    CK = C.dot(K)
    CKCt = C.dot(K.dot(C.T))

    assert np.allclose(C.sum(axis=1), 1, 1e-12)
    assert np.allclose(Z.sum(axis=1), 1, 1e-12)

    initial_cost = _kernel_aa_cost(K, Z, C, alpha)

    updated_Z = _update_kernel_aa_weights(
        Z, alpha, CK, CKCt)

    final_cost = _kernel_aa_cost(K, updated_Z, C, alpha)

    assert final_cost <= initial_cost
    assert np.allclose(updated_Z.sum(axis=1), 1, 1e-12)


def test_single_weights_update_reduces_cost_with_nonzero_delta():
    """Test single weights update reduces cost function with non-zero delta."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 50
    n_components = 5
    n_samples = 400
    delta = 0.5

    X = random_state.uniform(size=(n_samples, n_features))
    K = X.dot(X.T)

    C = right_stochastic_matrix(
        (n_components, n_samples), random_state=random_state)
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)
    alpha = random_state.uniform(low=(1 - delta), high=(1 + delta),
                                 size=(n_components,))

    CK = C.dot(K)
    CKCt = C.dot(K.dot(C.T))

    assert np.allclose(C.sum(axis=1), 1, 1e-12)
    assert np.allclose(Z.sum(axis=1), 1, 1e-12)

    initial_cost = _kernel_aa_cost(K, Z, C, alpha)

    updated_Z = _update_kernel_aa_weights(
        Z, alpha, CK, CKCt)

    final_cost = _kernel_aa_cost(K, updated_Z, C, alpha)

    assert final_cost <= initial_cost
    assert np.allclose(updated_Z.sum(axis=1), 1, 1e-12)


def test_exact_solution_with_zero_delta_is_weights_update_fixed_point():
    """Test exact solution for weights is fixed point of update step."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 30
    n_components = 10
    n_samples = 130
    tolerance = 1e-12

    basis = random_state.uniform(size=(n_components, n_features))

    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    archetype_indices = np.zeros(n_components, dtype='i8')
    for i in range(n_components):
        new_index = False
        current_index = 0

        while not new_index:
            new_index = True

            current_index = random_state.randint(
                low=0, high=n_samples)

            for index in archetype_indices:
                if current_index == index:
                    new_index = False

        archetype_indices[i] = current_index

    C = np.zeros((n_components, n_samples))
    component = 0
    for index in archetype_indices:
        C[component, index] = 1.0
        for i in range(n_components):
            if i == component:
                Z[index, i] = 1.0
            else:
                Z[index, i] = 0.0
        component += 1

    X = Z.dot(basis)
    basis_projection = C.dot(X)

    assert np.allclose(basis_projection, basis, tolerance)
    assert np.linalg.norm(X - Z.dot(C.dot(X))) < tolerance

    K = X.dot(X.T)

    alpha = np.ones((n_components,))

    CK = C.dot(K)
    CKCt = C.dot(K.dot(C.T))

    assert np.allclose(C.sum(axis=1), 1, 1e-12)
    assert np.allclose(Z.sum(axis=1), 1, 1e-12)

    initial_cost = _kernel_aa_cost(K, Z, C, alpha)

    updated_Z = _update_kernel_aa_weights(
        Z, alpha, CK, CKCt)

    final_cost = _kernel_aa_cost(K, updated_Z, C, alpha)

    assert abs(final_cost - initial_cost) < tolerance
    assert np.allclose(updated_Z.sum(axis=1), 1, 1e-12)
    assert np.allclose(updated_Z, Z, tolerance)


def test_repeated_weights_updates_converge_with_zero_delta():
    """Test repeated updates converge to a fixed point with delta = 0."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 10
    n_components = 3
    n_samples = 600
    max_iterations = 100
    tolerance = 1e-6

    X = random_state.uniform(size=(n_samples, n_features))
    K = X.dot(X.T)

    C = right_stochastic_matrix(
        (n_components, n_samples), random_state=random_state)
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(C.sum(axis=1), 1, tolerance)
    assert np.allclose(Z.sum(axis=1), 1, tolerance)

    delta = 0
    alpha = np.ones((n_components,))

    initial_cost = _kernel_aa_cost(K, Z, C, alpha)

    updated_Z, updated_C, updated_alpha, _, n_iter = _iterate_kernel_aa(
        K, Z, C, alpha, delta=delta,
        update_weights=True, update_dictionary=False,
        update_scale_factors=False,
        tolerance=tolerance, max_iterations=max_iterations,
        require_monotonic_cost_decrease=True)[:5]

    final_cost = _kernel_aa_cost(K, updated_Z, updated_C, updated_alpha)

    assert final_cost <= initial_cost
    assert n_iter < max_iterations

    assert np.allclose(updated_C, C, 1e-12)
    assert np.allclose(updated_alpha, alpha, 1e-12)
    assert np.allclose(updated_Z.sum(axis=1), 1, 1e-12)


def test_repeated_weights_updates_converge_with_nonzero_delta():
    """Test repeated updates converge to a fixed point with delta = 0."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_features = 30
    n_components = 11
    n_samples = 320
    max_iterations = 100
    tolerance = 1e-6

    X = random_state.uniform(size=(n_samples, n_features))
    K = X.dot(X.T)

    C = right_stochastic_matrix(
        (n_components, n_samples), random_state=random_state)
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assert np.allclose(C.sum(axis=1), 1, tolerance)
    assert np.allclose(Z.sum(axis=1), 1, tolerance)

    delta = 0.3
    alpha = random_state.uniform(low=(1 - delta), high=(1 + delta),
                                 size=(n_components,))

    initial_cost = _kernel_aa_cost(K, Z, C, alpha)

    updated_Z, updated_C, updated_alpha, _, n_iter = _iterate_kernel_aa(
        K, Z, C, alpha, delta=delta,
        update_weights=True, update_dictionary=False,
        update_scale_factors=False,
        tolerance=tolerance, max_iterations=max_iterations,
        require_monotonic_cost_decrease=True)[:5]

    final_cost = _kernel_aa_cost(K, updated_Z, updated_C, updated_alpha)

    assert final_cost <= initial_cost
    assert n_iter < max_iterations

    assert np.allclose(updated_C, C, 1e-12)
    assert np.allclose(updated_alpha, alpha, 1e-12)
    assert np.allclose(updated_Z.sum(axis=1), 1, 1e-12)


def test_finds_elements_of_3_point_convex_hull():
    """Test finds archetypes in convex hull for 2D example."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 50
    n_components = 3
    max_iterations = 500
    tolerance = 1e-6

    basis = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    expected_Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assignments = np.array([5, 27, 32])
    for i in range(n_components):
        expected_Z[assignments[i]] = np.zeros(n_components)
        expected_Z[assignments[i], i] = 1

    X = expected_Z.dot(basis)
    K = X.dot(X.T)

    C = right_stochastic_matrix(
        (n_components, n_samples), random_state=random_state)
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)
    alpha = np.ones((n_components,))

    assert np.allclose(C.sum(axis=1), 1, 1e-12)
    assert np.allclose(Z.sum(axis=1), 1, 1e-12)

    delta = 0

    aa = KernelAA(n_components=n_components, delta=delta, init='custom',
                  max_iterations=max_iterations, tolerance=tolerance)

    solution_Z = aa.fit_transform(K, dictionary=C, weights=Z, alpha=alpha)
    solution_C = aa.dictionary

    assert aa.n_iter < max_iterations

    assert np.allclose(solution_C.sum(axis=1), 1, 1e-12)
    assert np.allclose(solution_Z.sum(axis=1), 1, 1e-12)

    main_components = solution_C.argmax(axis=1)
    main_components = sorted(main_components)
    for i in range(n_components):
        assert main_components[i] == assignments[i]


def test_finds_elements_of_4_point_convex_hull():
    """Test finds archetypes in convex hull for 3D example."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 123
    n_components = 4
    max_iter = 500
    tolerance = 1e-12

    basis = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

    expected_Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)

    assignments = np.array([8, 9, 56, 90])
    for i in range(n_components):
        expected_Z[assignments[i]] = np.zeros(n_components)
        expected_Z[assignments[i], i] = 1

    expected_C = np.zeros((n_components, n_samples), dtype='f8')
    for i in range(n_components):
        expected_C[i, assignments[i]] = 1

    X = expected_Z.dot(basis)

    assert np.linalg.norm(X - expected_Z.dot(expected_C.dot(X))) < tolerance

    K = X.dot(X.T)

    C = right_stochastic_matrix(
        (n_components, n_samples), random_state=random_state)
    Z = right_stochastic_matrix(
        (n_samples, n_components), random_state=random_state)
    alpha = np.ones((n_components,))

    assert np.allclose(C.sum(axis=1), 1, 1e-12)
    assert np.allclose(Z.sum(axis=1), 1, 1e-12)

    delta = 0
    aa = KernelAA(n_components=n_components, delta=delta, init='custom',
                  max_iterations=max_iter, tolerance=tolerance)

    solution_Z = aa.fit_transform(K, dictionary=C, weights=Z, alpha=alpha)
    solution_C = aa.dictionary

    assert aa.n_iter < max_iter

    assert np.allclose(solution_C.sum(axis=1), 1, 1e-12)
    assert np.allclose(solution_Z.sum(axis=1), 1, 1e-12)

    main_components = solution_C.argmax(axis=1)
    main_components = sorted(main_components)
    for i in range(n_components):
        assert main_components[i] == assignments[i]
