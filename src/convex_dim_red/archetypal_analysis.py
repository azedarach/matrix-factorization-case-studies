"""
Provides routines for performing archetypal analysis.
"""

# License: MIT

from __future__ import absolute_import, division, print_function

import numbers
import time
import warnings

import numpy as np
from numba import guvectorize, jit, float64, int32

from sklearn.utils import check_array, check_random_state

from .furthest_sum import furthest_sum
from .simplex_projection import simplex_project_rows
from .spg import quad_simplex_spg, spg
from .stochastic_matrices import right_stochastic_matrix
from .validation_utils import check_array_shape, check_stochastic_matrix


INTEGER_TYPES = (numbers.Integral, np.integer)

INITIALIZATION_METHODS = (None, 'random', 'furthest_sum',)


def _check_init_weights(weights, shape, whom):

    weights = check_array(weights)
    check_stochastic_matrix(weights, shape, whom, axis=1)


def _check_init_dictionary(dictionary, shape, whom):

    dictionary = check_array(dictionary)
    check_stochastic_matrix(dictionary, shape, whom, axis=1)


def _check_init_scale_factors(alpha, delta, shape, whom):

    check_array_shape(alpha, shape, whom)

    if np.any(np.logical_or(alpha < 1 - delta, alpha > 1 + delta)):
        raise ValueError(
            'Initial scale factors infeasible in %s' % (whom))


def _initialize_kernel_aa_dictionary_random(
        kernel, n_components, random_state=None):

    rng = check_random_state(random_state)

    n_samples = kernel.shape[0]

    return right_stochastic_matrix((n_components, n_samples),
                                   random_state=rng)


def _initialize_kernel_aa_weights_random(
        kernel, n_components, random_state=None):

    rng = check_random_state(random_state)

    n_samples = kernel.shape[0]

    return right_stochastic_matrix((n_samples, n_components),
                                   random_state=rng)


def _initialize_kernel_aa_scale_factors_random(
        n_components, delta=0, random_state=None):

    rng = check_random_state(random_state)

    if delta != 0:
        return rng.uniform(low=(1 - delta), high=(1+delta), size=(n_components,))

    return np.ones(n_components)


def _initialize_kernel_aa_dictionary_furthest_sum(
        kernel, n_components, start_index=None, n_extra_steps=10,
        exclude=None, random_state=None):
    rng = check_random_state(random_state)

    n_samples = kernel.shape[0]
    if start_index is None:
        start_index = rng.randint(n_samples)

    if exclude is None:
        exclude = np.array([], dtype='i8')

    kernel_diag = np.diag(kernel)
    dissimilarities = np.sqrt(
        np.tile(kernel_diag, (n_samples, 1)) -
        2 * kernel +
        np.tile(kernel_diag[:, np.newaxis], (1, n_samples)))

    selected = furthest_sum(
        dissimilarities, n_components, start_index, exclude, n_extra_steps)

    dictionary = np.zeros((n_components, n_samples),
                          dtype=kernel.dtype)
    for i in range(n_components):
        dictionary[i, selected[i]] = 1

    return dictionary


def _initialize_kernel_aa_dictionary(kernel, n_components, init='furthest_sum',
                                     random_state=None, **kwargs):
    if init is None:
        init = 'furthest_sum'

    if init == 'furthest_sum':
        start_index = kwargs.get('start_index', None)
        n_extra_steps = kwargs.get('n_extra_steps', 10)
        exclude = kwargs.get('exclude', None)

        return _initialize_kernel_aa_dictionary_furthest_sum(
            kernel, n_components, start_index=start_index,
            n_extra_steps=n_extra_steps,
            exclude=exclude, random_state=random_state)

    if init == 'random':
        return _initialize_kernel_aa_dictionary_random(
            kernel, n_components, random_state=random_state)

    raise ValueError(
        'Invalid init parameter: got %r instead of one of %r' %
        (init, INITIALIZATION_METHODS))


def _initialize_kernel_aa_weights(kernel, n_components, init='furthest_sum',
                                  random_state=None):
    if init is None:
        init = 'furthest_sum'

    if init in ('furthest_sum', 'random'):
        return _initialize_kernel_aa_weights_random(
            kernel, n_components, random_state=random_state)

    raise ValueError(
        'Invalid init parameter: got %r instead of one of %r' %
        (init, INITIALIZATION_METHODS))


def _initialize_kernel_aa(kernel, n_components, init='furthest_sum',
                          random_state=None, **kwargs):
    if init is None:
        init = 'furthest_sum'

    rng = check_random_state(random_state)

    dictionary = _initialize_kernel_aa_dictionary(
        kernel, n_components, init=init, random_state=rng, **kwargs)

    weights = _initialize_kernel_aa_weights(
        kernel, n_components, init=init, random_state=rng, **kwargs)

    return dictionary, weights


def _check_if_cost_increased(old, new, tolerance, stage, require_decrease=True):
    """Check if cost has increased significantly compared to tolerance."""

    cost_increased = (new > old) and (abs(new - old) > tolerance)

    if cost_increased and require_decrease:
        raise RuntimeError(
            'factorization cost increased after {} update'.format(stage))


def _get_stopping_criteria(stopping_criterion):
    """Construct stopping criterion."""

    if stopping_criterion not in ('abs_delta_f', 'rel_delta_f'):
        raise ValueError(
            "unsupported stopping criterion '%s'" % stopping_criterion)

    if stopping_criterion == 'abs_delta_f':

        def has_converged(old_cost, new_cost, tolerance):
            cost_delta = new_cost - old_cost
            return abs(cost_delta) < tolerance

    elif stopping_criterion == 'rel_delta_f':

        def has_converged(old_cost, new_cost, tolerance):
            cost_delta = new_cost - old_cost
            max_cost = max(abs(new_cost), abs(old_cost))
            return abs(cost_delta / max_cost) < tolerance

    return has_converged


@jit(nopython=True)
def _kernel_aa_cost(K, weights, dictionary, alpha):
    """Evaluate kernel AA cost function."""

    n_samples = K.shape[0]

    da = np.diag(alpha)

    CK = dictionary.dot(K)
    CKCt = CK.dot(dictionary.T)
    CKZ = CK.dot(weights)
    ZtZ = weights.T.dot(weights)

    trace_K = np.trace(K)
    trace_DCKZ = np.trace(da.dot(CKZ))
    trace_DZtZDCKCt = np.trace((da.dot(ZtZ.dot(da))).dot(CKCt))

    return 0.5 * (trace_K - 2 * trace_DCKZ + trace_DZtZDCKCt) / n_samples


@jit(nopython=True)
def _kernel_aa_scale_factors_objective(alpha, trace_K, CKZ, ZtZ, CKCt):
    """Evaluate kernel AA cost function with fixed weights and dictionary."""

    n_samples = CKZ.shape[1]

    a2 = np.outer(alpha, alpha)

    return 0.5 * (trace_K - 2 * alpha.dot(np.diag(CKZ)) +
                  np.sum(a2 * ZtZ * CKCt)) / n_samples


@jit(nopython=True)
def _kernel_aa_scale_factors_gradient(alpha, CKZ, ZtZ, CKCt):
    """Evaluate gradient of cost function with fixed weights and dictionary."""

    n_samples = CKZ.shape[1]

    da = np.diag(alpha)

    return np.diag(ZtZ.dot(da.dot(CKCt)) - CKZ) / n_samples


def _update_kernel_aa_scale_factors(alpha, trace_K, CKZ, ZtZ, CKCt, delta,
                                    **kwargs):
    """Update kernel AA scale factors."""

    def f(x):
        return _kernel_aa_scale_factors_objective(x, trace_K, CKZ, ZtZ, CKCt)

    def df(x):
        return _kernel_aa_scale_factors_gradient(x, CKZ, ZtZ, CKCt)

    def project(x):
        return np.fmin(np.fmax(1.0 - delta, x), 1.0 + delta)

    alpha, _, _, _ = spg(f, df, alpha, project=project, **kwargs)

    return alpha


@jit(nopython=True)
def _aa_dictionary_cost(X, dictionary, trace_XXt, XXtZD, DZtZD):
    """Evaluate cost function with fixed weights and scale factors."""

    n_samples = dictionary.shape[0]
    CX = dictionary.dot(X)

    return (0.5 * (trace_XXt - 2 * np.trace(dictionary.dot(XXtZD)) +
                   np.trace(DZtZD.dot(CX.dot(CX.T)))) /
            n_samples)


@jit(nopython=True)
def _kernel_aa_dictionary_cost(K, dictionary, trace_K, KZD, DZtZD):
    """Evaluate cost function with fixed weights and scale factors."""

    n_samples = dictionary.shape[0]

    return (0.5 * (trace_K - 2 * np.trace(dictionary.dot(KZD)) +
                   np.trace(DZtZD.dot(dictionary.dot(K.dot(dictionary.T))))) /
            n_samples)


@jit(nopython=True)
def _kernel_aa_dictionary_gradient(K, dictionary, KZD, DZtZD):
    """Evaluate gradient of cost function with fixed weights and scale factors."""

    n_samples = dictionary.shape[0]

    return (DZtZD.dot(dictionary.dot(K)) - KZD.T) / n_samples


@jit(nopython=True)
def _aa_dictionary_gradient(X, dictionary, XXtZD, DZtZD):
    """Evaluate gradient of cost function with fixed weights and scale factors."""

    n_samples = dictionary.shape[1]

    CX = dictionary.dot(X)

    return (DZtZD.dot(CX.dot(X.T)) - XXtZD.T) / n_samples


def _update_kernel_aa_dictionary(K, dictionary, alpha, trace_K, KZ, ZtZ, **kwargs):
    """Update dictionary for kernel AA."""

    da = np.diag(alpha)

    KZD = KZ.dot(da)
    DZtZD = da.dot(ZtZ.dot(da))

    def f(x):
        return _kernel_aa_dictionary_cost(K, x, trace_K, KZD, DZtZD)

    def df(x):
        return _kernel_aa_dictionary_gradient(K, x, KZD, DZtZD)

    dictionary, _, _, _ = spg(f, df, dictionary, project=simplex_project_rows,
                              **kwargs)

    return dictionary


def _update_aa_dictionary(X, dictionary, alpha, trace_XXt, XXtZ, ZtZ, **kwargs):
    """Update dictionary for AA."""

    da = np.diag(alpha)

    XXtZD = XXtZ.dot(da)
    DZtZD = da.dot(ZtZ.dot(da))

    def f(x):
        return _aa_dictionary_cost(X, x, trace_XXt, XXtZD, DZtZD)

    def df(x):
        return _aa_dictionary_gradient(X, x, XXtZD, DZtZD)

    dictionary, _, _, _ = spg(f, df, dictionary, project=simplex_project_rows,
                              **kwargs)

    return dictionary


@guvectorize(
    [(float64[:, :], float64[:, :], float64[:, :],
      float64, int32, float64, float64, float64, float64, float64, float64,
      float64, float64, int32, int32, float64[:, :])],
    '(k, k), (k, i), (i, k), (), (), (), (), (), (), (), (), (), (), (), () -> (i, k)',
    nopython=True, target='parallel')
def _gu_update_kernel_aa_weights(CKCt, CK, initial_weights,
                                 gamma, memory,
                                 sigma_one, sigma_two, lambda_min,
                                 alpha0, alpha_min, alpha_max,
                                 epsilon_one, epsilon_two,
                                 max_iterations, max_feval, final_weights):

    n_samples = initial_weights.shape[0]

    for t in range(n_samples):

        final_weights[t] = quad_simplex_spg(
            CKCt, -CK[:, t], initial_weights[t], gamma=gamma, memory=memory,
            sigma_one=sigma_one, sigma_two=sigma_two, lambda_min=lambda_min,
            alpha0=alpha0, alpha_min=alpha_min, alpha_max=alpha_max,
            epsilon_one=epsilon_one, epsilon_two=epsilon_two,
            max_iterations=max_iterations, max_feval=max_feval)


def _update_kernel_aa_weights(weights, alpha, CK, CKCt, **solver_kwargs):
    """Update weights for kernel AA."""

    gamma = solver_kwargs.get('gamma', 1e-4)
    memory = solver_kwargs.get('memory', 1)
    sigma_one = solver_kwargs.get('sigma_one', 0.1)
    sigma_two = solver_kwargs.get('sigma_two', 0.9)
    lambda_min = solver_kwargs.get('lambda_min', 1e-10)
    alpha0 = solver_kwargs.get('alpha0', -1.0)
    alpha_min = solver_kwargs.get('alpha_min', 1e-5)
    alpha_max = solver_kwargs.get('alpha_max', 1e3)
    epsilon_one = solver_kwargs.get('epsilon_one', 1e-10)
    epsilon_two = solver_kwargs.get('epsilon_two', 1e-6)
    max_iterations = solver_kwargs.get('max_iterations', 1000)
    max_feval = solver_kwargs.get('max_feval', 2000)

    da = np.diag(alpha)

    CKCt = da.dot(CKCt.dot(da))
    CK = da.dot(CK)

    # pylint: disable=no-value-for-parameter
    return _gu_update_kernel_aa_weights(
        CKCt, CK, weights, gamma, memory,
        sigma_one, sigma_two, lambda_min,
        alpha0, alpha_min, alpha_max,
        epsilon_one, epsilon_two,
        max_iterations, max_feval)


def _iterate_kernel_aa(K, weights, dictionary, alpha, delta=0,
                       update_weights=True, update_dictionary=True,
                       update_scale_factors=True, tolerance=1e-6,
                       max_iterations=1000, verbose=0, **kwargs):
    """Iteratively update kernel AA parameters until convergence is reached."""

    n_samples, n_components = weights.shape

    # Pre-compute constants.
    da = np.diag(alpha)
    ZtZ = weights.T.dot(weights)
    CK = dictionary.dot(K)
    CKCt = CK.dot(dictionary.T)
    KZ = K.dot(weights)
    CKZ = dictionary.dot(KZ)

    trace_K = K.trace()
    trace_DCKZ = da.dot(CKZ).trace()
    trace_DZtZDCKCt = (da.dot(ZtZ.dot(da))).dot(CKCt).trace()

    new_cost = (0.5 * (trace_K - 2 * trace_DCKZ + trace_DZtZDCKCt) / n_samples)

    old_cost = None

    # Determine if the cost decrease should be required to be strictly
    # monotonic.
    require_monotonic_cost_decrease = kwargs.get(
        'require_monotonic_cost_decrease', True)

    # Define stopping criterion.
    stopping_criterion = kwargs.get('stopping_criterion', 'abs_delta_f')
    has_converged = _get_stopping_criteria(stopping_criterion)

    # Get additional configuration parameters for solvers.
    dictionary_solver_kwargs = kwargs.get('dictionary_solver_kwargs', {})
    weights_solver_kwargs = kwargs.get('weights_solver_kwargs', {})
    scale_factors_solver_kwargs = kwargs.get('scale_factors_solver_kwargs', {})

    # Iterate until stopping criteria are satisfied or maximum number of
    # iterations is reached.
    iter_times = []
    cost_deltas = []

    if verbose:
        print("*** Kernel AA: n_components = {:d} ***".format(
            n_components))
        print('{:<12s} | {:<13s} | {:<13s} | {:<12s}'.format(
            'Iteration', 'Cost', 'Cost delta', 'Time'))
        print(80 * '-')

    for n_iter in range(max_iterations):

        start_time = time.perf_counter()

        old_cost = new_cost

        if update_scale_factors and delta != 0:

            # Find optimal scale factors for fixed weights and dictionary.
            alpha = _update_kernel_aa_scale_factors(
                alpha, trace_K, CKZ, ZtZ, CKCt, delta,
                **scale_factors_solver_kwargs)

            da = np.diag(alpha)

            trace_DCKZ = da.dot(CKZ).trace()
            trace_DZtZDCKCt = (da.dot(ZtZ.dot(da))).dot(CKCt).trace()

            new_cost = (0.5 * (trace_K - 2 * trace_DCKZ + trace_DZtZDCKCt) /
                        n_samples)

            _check_if_cost_increased(
                old_cost, new_cost, tolerance, 'scale factors',
                require_decrease=require_monotonic_cost_decrease)

        if update_dictionary:

            # Find optimal dictionary for fixed weights and scale factors.
            dictionary = _update_kernel_aa_dictionary(
                K, dictionary, alpha, trace_K, KZ, ZtZ,
                **dictionary_solver_kwargs)

            CK = dictionary.dot(K)
            CKCt = CK.dot(dictionary.T)
            CKZ = dictionary.dot(KZ)

            trace_DCKZ = da.dot(CKZ).trace()
            trace_DZtZDCKCt = (da.dot(ZtZ.dot(da))).dot(CKCt).trace()

            new_cost = (0.5 * (trace_K - 2 * trace_DCKZ + trace_DZtZDCKCt) /
                        n_samples)

            _check_if_cost_increased(
                old_cost, new_cost, tolerance, 'dictionary',
                require_decrease=require_monotonic_cost_decrease)

        if update_weights:

            # Find optimal weights for fixed scale factors and dictionary.
            weights = _update_kernel_aa_weights(
                weights, alpha, CK, CKCt,
                **weights_solver_kwargs)

            ZtZ = weights.T.dot(weights)
            KZ = K.dot(weights)
            CKZ = dictionary.dot(KZ)

            trace_DCKZ = da.dot(CKZ).trace()
            trace_DZtZDCKCt = (da.dot(ZtZ.dot(da))).dot(CKCt).trace()

            new_cost = (0.5 * (trace_K - 2 * trace_DCKZ + trace_DZtZDCKCt) / n_samples)

            _check_if_cost_increased(
                old_cost, new_cost, tolerance, 'weights',
                require_decrease=require_monotonic_cost_decrease)

        end_time = time.perf_counter()

        iter_times.append(end_time - start_time)
        cost_deltas.append(new_cost - old_cost)

        if verbose:
            print('{:12d} | {: 12.6e} | {: 12.6e} | {: 12.6e}'.format(
                n_iter + 1, new_cost, new_cost - old_cost, end_time - start_time))

        if has_converged(old_cost, new_cost, tolerance):
            if verbose:
                print('*** Converged at iteration {:d} ***'.format(
                    n_iter + 1))
            break

    return (weights, dictionary, alpha, new_cost, n_iter,
            np.mean(iter_times), cost_deltas)


def _iterate_aa(X, weights, dictionary, alpha, delta=0,
                update_weights=True, update_dictionary=True,
                update_scale_factors=True, tolerance=1e-6,
                max_iterations=1000, verbose=0, **kwargs):
    """Iteratively update AA parameters until convergence is reached."""

    n_samples, n_components = weights.shape

    # Pre-compute constants.
    da = np.diag(alpha)
    ZtZ = weights.T.dot(weights)
    CX = dictionary.dot(X)
    CXXt = CX.dot(X.T)
    CXXtCt = CX.dot(CX.T)
    XtZ = X.T.dot(weights)
    XXtZ = X.dot(XtZ)
    CXXtZ = dictionary.dot(XXtZ)

    trace_XXt = np.trace(X.dot(X.T))
    trace_DCXXtZ = da.dot(CXXtZ).trace()
    trace_DZtZDCXXtCt = (da.dot(ZtZ.dot(da))).dot(CXXtCt).trace()

    new_cost = (0.5 * (trace_XXt - 2 * trace_DCXXtZ + trace_DZtZDCXXtCt) / n_samples)

    old_cost = None

    # Determine if the cost decrease should be required to be strictly
    # monotonic.
    require_monotonic_cost_decrease = kwargs.get(
        'require_monotonic_cost_decrease', True)

    # Define stopping criterion.
    stopping_criterion = kwargs.get('stopping_criterion', 'abs_delta_f')
    has_converged = _get_stopping_criteria(stopping_criterion)

    # Get additional configuration parameters for solvers.
    dictionary_solver_kwargs = kwargs.get('dictionary_solver_kwargs', {})
    weights_solver_kwargs = kwargs.get('weights_solver_kwargs', {})
    scale_factors_solver_kwargs = kwargs.get('scale_factors_solver_kwargs', {})

    # Iterate until stopping criteria are satisfied or maximum number of
    # iterations is reached.
    iter_times = []
    cost_deltas = []

    if verbose:
        print("*** AA: n_components = {:d} ***".format(
            n_components))
        print('{:<12s} | {:<13s} | {:<13s} | {:<12s}'.format(
            'Iteration', 'Cost', 'Cost delta', 'Time'))
        print(80 * '-')

    for n_iter in range(max_iterations):

        start_time = time.perf_counter()

        old_cost = new_cost

        if update_scale_factors and delta != 0:

            # Find optimal scale factors for fixed weights and dictionary.
            alpha = _update_kernel_aa_scale_factors(
                alpha, trace_XXt, CXXtZ, ZtZ, CXXtCt, delta,
                **scale_factors_solver_kwargs)

            da = np.diag(alpha)

            trace_DCXXtZ = da.dot(CXXtZ).trace()
            trace_DZtZDCXXtCt = (da.dot(ZtZ.dot(da))).dot(CXXtCt).trace()

            new_cost = (0.5 * (trace_XXt - 2 * trace_DCXXtZ + trace_DZtZDCXXtCt) /
                        n_samples)

            _check_if_cost_increased(
                old_cost, new_cost, tolerance, 'scale factors',
                require_decrease=require_monotonic_cost_decrease)

        if update_dictionary:

            # Find optimal dictionary for fixed weights and scale factors.
            dictionary = _update_aa_dictionary(
                X, dictionary, alpha, trace_XXt, XXtZ, ZtZ,
                **dictionary_solver_kwargs)

            CX = dictionary.dot(X)
            CXXt = CX.dot(X.T)
            CXXtCt = CX.dot(CX.T)
            CXXtZ = dictionary.dot(XXtZ)

            trace_DCXXtZ = da.dot(CXXtZ).trace()
            trace_DZtZDCXXtCt = (da.dot(ZtZ.dot(da))).dot(CXXtCt).trace()

            new_cost = (0.5 * (trace_XXt - 2 * trace_DCXXtZ + trace_DZtZDCXXtCt) /
                        n_samples)

            _check_if_cost_increased(
                old_cost, new_cost, tolerance, 'dictionary',
                require_decrease=require_monotonic_cost_decrease)

        if update_weights:

            # Find optimal weights for fixed scale factors and dictionary.
            weights = _update_kernel_aa_weights(
                weights, alpha, CXXt, CXXtCt,
                **weights_solver_kwargs)

            ZtZ = weights.T.dot(weights)
            XtZ = X.T.dot(weights)
            XXtZ = X.dot(XtZ)
            CXXtZ = dictionary.dot(XXtZ)

            trace_DCXXtZ = da.dot(CXXtZ).trace()
            trace_DZtZDCXXtCt = (da.dot(ZtZ.dot(da))).dot(CXXtCt).trace()

            new_cost = (0.5 * (trace_XXt - 2 * trace_DCXXtZ + trace_DZtZDCXXtCt) / n_samples)

            _check_if_cost_increased(
                old_cost, new_cost, tolerance, 'weights',
                require_decrease=require_monotonic_cost_decrease)

        end_time = time.perf_counter()

        iter_times.append(end_time - start_time)
        cost_deltas.append(new_cost - old_cost)

        if verbose:
            print('{:12d} | {: 12.6e} | {: 12.6e} | {: 12.6e}'.format(
                n_iter + 1, new_cost, new_cost - old_cost, end_time - start_time))

        if has_converged(old_cost, new_cost, tolerance):
            if verbose:
                print('*** Converged at iteration {:d} ***'.format(
                    n_iter + 1))
            break

    return (weights, dictionary, alpha, new_cost, n_iter,
            np.mean(iter_times), cost_deltas)


class KernelAA():
    """Kernel archetypal analysis.

    Performs archetypal analysis given a kernel matrix computed
    from the original data.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of archetypes. If None, then all
        samples are used.

    delta : float, default: 0
        Relaxation parameter for the dictionary.

    init : None | 'random' | 'furthest_sum' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'furthest_sum'

        - 'random': dictionary and weights are initialized to
          random stochastic matrices.

        - 'furthest_sum': dictionary is initialized using FurthestSum
          method, and weights are initialized to a random stochastic
          matrix.

        - 'custom': use custom matrices for dictionary and weights.

    tolerance: float, default: 1e-6
        Tolerance of the stopping condition.

    max_iterations : integer, default: 1000
        Maximum number of iterations before stopping.

    verbose : integer, default: 0
        The verbosity level.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    Attributes
    ----------
    dictionary_ : array-like, shape (n_components, n_samples)
        The dictionary containing the composition of the archetypes.

    cost_ : number
        Value of the cost function for the obtained factorization.

    n_iter_ : integer
        Actual number of iterations.

    Examples
    --------
    import numpy as np
    X = np.random.rand(10, 4)
    K = np.dot(X, X.T)
    from archetypal_analysis import KernelAA
    model = KernelAA(n_components=2, init='furthest_sum', random_state=0)
    weights = model.fit_transform(K)
    dictionary = model.dictionary_

    References
    ----------
    M. Morup and L. K. Hansen, "Archetypal analysis for machine learning
    and data mining", Neurocomputing 80 (2012) 54 - 63.
    """

    def __init__(self, n_components, delta=0, init=None,
                 tolerance=1e-6, max_iterations=1000, verbose=0,
                 random_state=None, **kwargs):
        self.n_components = n_components
        self.delta = delta
        self.init = init
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.require_monotonic_cost_decrease = kwargs.get(
            'require_monotonic_cost_decrease', True)
        self.stopping_criterion = kwargs.get('stopping_criterion', 'abs_delta_f')

        self.weights = None
        self.dictionary = None
        self.alpha = None
        self.cost = 0
        self.n_iter = 0
        self.avg_time_per_iter = 0
        self.cost_deltas = None

        self.weights_solver_kwargs = kwargs.get('weights_solver_kwargs', {})
        self.dictionary_solver_kwargs = kwargs.get('dictionary_solver_kwargs', {})
        self.scale_factors_solver_kwargs = kwargs.get('scale_factors_solver_kwargs', {})

    def _kernel_aa(self, kernel, dictionary=None, weights=None, alpha=None,
                   update_dictionary=True, update_weights=True,
                   update_scale_factors=True, **kwargs):
        """Perform kernel archetypal analysis."""

        n_samples = kernel.shape[0]

        if kernel.shape[1] != n_samples:
            raise ValueError(
                'Expected square kernel matrix in %s. '
                'Got shape %s' % ('kernel_aa', kernel.shape))

        if self.n_components is None:
            self.n_components = n_samples

        if not isinstance(self.n_components, INTEGER_TYPES) or self.n_components <= 0:
            raise ValueError('Number of components must be a positive integer;'
                             ' got (n_components=%r)' % self.n_components)
        if not isinstance(self.max_iterations, INTEGER_TYPES) or self.max_iterations <= 0:
            raise ValueError('Maximum number of iterations must be a positive '
                             'integer; got (max_iterations=%r)' % self.max_iterations)
        if not isinstance(self.tolerance, numbers.Number) or self.tolerance < 0:
            raise ValueError('Tolerance for stopping criteria must be '
                             'positive; got (tolerance=%r)' % self.tolerance)

        if self.init == 'custom':
            _check_init_weights(weights, (n_samples, self.n_components),
                                '_kernel_aa (input weights)')
            _check_init_dictionary(dictionary, (self.n_components, n_samples),
                                   '_kernel_aa (input dictionary)')
            _check_init_scale_factors(alpha, self.delta, (self.n_components,),
                                      '_kernel_aa (input scale factors)')
        elif not update_dictionary and update_weights:
            _check_init_dictionary(dictionary, (self.n_components, n_samples),
                                   '_kernel_aa (input dictionary)')
            weights = _initialize_kernel_aa_weights(
                kernel, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)
        elif update_dictionary and not update_weights:
            _check_init_weights(weights, (n_samples, self.n_components),
                                '_kernel_aa (input weights)')
            dictionary = _initialize_kernel_aa_dictionary(
                kernel, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)
        else:
            dictionary, weights = _initialize_kernel_aa(
                kernel, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)

        if alpha is None:
            alpha = _initialize_kernel_aa_scale_factors_random(
                self.n_components, delta=self.delta,
                random_state=self.random_state)

        else:
            _check_init_scale_factors(alpha, self.delta, (self.n_components,),
                                      '_kernel_aa (input scale factors)')

        self.weights = weights.copy()
        self.dictionary = dictionary.copy()
        self.alpha = alpha.copy()

        if kernel.dtype != self.weights.dtype:
            kernel = kernel.astype(self.weights.dtype)

        self.weights, self.dictionary, self.alpha, cost, n_iter, avg_time_per_iter, cost_deltas = \
            _iterate_kernel_aa(
                kernel, self.weights, self.dictionary, self.alpha, delta=self.delta,
                update_weights=update_weights,
                update_dictionary=update_dictionary,
                update_scale_factors=update_scale_factors,
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
                require_monotonic_cost_decrease=self.require_monotonic_cost_decrease,
                stopping_criterion=self.stopping_criterion,
                weights_solver_kwargs=self.weights_solver_kwargs,
                dictionary_solver_kwargs=self.dictionary_solver_kwargs,
                scale_factors_solver_kwargs=self.scale_factors_solver_kwargs)

        if n_iter == self.max_iterations and self.tolerance > 0:
            warnings.warn('Maximum number of iterations %d reached.' %
                          self.max_iterations, UserWarning)

        return cost, n_iter, avg_time_per_iter, cost_deltas

    def fit_transform(self, data, dictionary=None, weights=None, alpha=None,
                      **kwargs):
        """Perform kernel archetypal analysis and return transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_samples)
            Kernel matrix to be factorized.

        dictionary : array-like, shape (n_components, n_samples)
            If init='custom', used as initial guess for solution.

        weights : array-like, shape (n_samples, n_components)
            If init='custom', used as initial guess for solution.

        alpha : array-like, shape (n_components,)
            If init='custom', used as initial guess for solution.

        Returns
        -------
        weights : array-like, shape (n_samples, n_components)
            Representation of the data.
        """

        cost_, n_iter_, avg_time_per_iter_, cost_deltas_ = self._kernel_aa(
            data,
            dictionary=dictionary,
            weights=weights, alpha=alpha, **kwargs)

        self.cost = cost_
        self.n_iter = n_iter_
        self.avg_time_per_iter = avg_time_per_iter_
        self.cost_deltas = cost_deltas_

        return self.weights

    def fit(self, kernel, **kwargs):
        """Perform kernel archetypal analysis on given kernel.

        Parameters
        ----------
        kernel : array-like, shape (n_samples, n_samples)
            Kernel matrix to perform analysis on.

        Returns
        -------
        self
        """
        self.fit_transform(kernel, **kwargs)
        return self


class ArchetypalAnalysis():
    """Standard archetypal analysis.

    Performs archetypal analysis by minimizing the cost function::

        ||X - Z C X||_Fro^2

    by performing a series of alternating minimizations with
    respect to the dictionary C and weights Z.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of archetypes. If None, then all
        samples are used.

    delta : float, default: 0
        Relaxation parameter for the dictionary.

    init : None | 'random' | 'furthest_sum' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'furthest_sum'

        - 'random': dictionary and weights are initialized to
          random stochastic matrices.

        - 'furthest_sum': dictionary is initialized using FurthestSum
          method, and weights are initialized to a random stochastic
          matrix.

        - 'custom': use custom matrices for dictionary and weights.

    tolerance: float, default: 1e-6
        Tolerance of the stopping condition.

    max_iterations : integer, default: 1000
        Maximum number of iterations before stopping.

    verbose : integer, default: 0
        The verbosity level.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    Attributes
    ----------
    dictionary : array-like, shape (n_components, n_samples)
        The dictionary containing the composition of the archetypes.

    archetypes : array-like, shape (n_components, n_features)
        The fitted archetypes.

    cost : number
        Value of the cost function for the obtained factorization.

    n_iter : integer
        Actual number of iterations.

    avg_time_per_iter : float
        Average time required for each iteration.

    cost_delta : array
        Change in cost function at each iteration.

    Examples
    --------
    import numpy as np
    X = np.random.rand(4, 10)
    from archetypal_analysis import ArchetypalAnalysis
    model = ArchetypalAnalysis(n_components=2, init='furthest_sum',
                               random_state=0)
    weights = model.fit_transform(X)
    dictionary = model.dictionary

    References
    ----------
    M. Morup and L. K. Hansen, "Archetypal analysis for machine learning
    and data mining", Neurocomputing 80 (2012) 54 - 63.
    """
    def __init__(self, n_components, delta=0, init=None,
                 tolerance=1e-6, max_iterations=1000, verbose=0,
                 random_state=None, **kwargs):
        self.n_components = n_components
        self.delta = delta
        self.init = init
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.require_monotonic_cost_decrease = kwargs.get(
            'require_monotonic_cost_decrease', True)
        self.stopping_criterion = kwargs.get('stopping_criterion', 'abs_delta_f')

        self.weights = None
        self.dictionary = None
        self.alpha = None
        self.cost = 0
        self.n_iter = 0
        self.avg_time_per_iter = 0
        self.cost_deltas = None

        self.weights_solver_kwargs = kwargs.get('weights_solver_kwargs', {})
        self.dictionary_solver_kwargs = kwargs.get('dictionary_solver_kwargs', {})
        self.scale_factors_solver_kwargs = kwargs.get('scale_factors_solver_kwargs', {})
        self.archetypes = None

    def _aa(self, data, dictionary=None, weights=None, alpha=None,
            update_dictionary=True, update_weights=True,
            update_scale_factors=True, **kwargs):
        """Perform archetypal analysis."""

        n_samples = data.shape[0]
        kernel = data.dot(data.T)

        if self.n_components is None:
            self.n_components = data.shape[1]

        if not isinstance(self.n_components, INTEGER_TYPES) or self.n_components <= 0:
            raise ValueError('Number of components must be a positive integer;'
                             ' got (n_components=%r)' % self.n_components)
        if not isinstance(self.max_iterations, INTEGER_TYPES) or self.max_iterations <= 0:
            raise ValueError('Maximum number of iterations must be a positive '
                             'integer; got (max_iterations=%r)' % self.max_iterations)
        if not isinstance(self.tolerance, numbers.Number) or self.tolerance < 0:
            raise ValueError('Tolerance for stopping criteria must be '
                             'positive; got (tolerance=%r)' % self.tolerance)

        if self.init == 'custom':
            _check_init_weights(weights, (n_samples, self.n_components),
                                '_aa (input weights)')
            _check_init_dictionary(dictionary, (self.n_components, n_samples),
                                   '_aa (input dictionary)')
            _check_init_scale_factors(alpha, self.delta, (self.n_components,),
                                      '_aa (input scale factors)')
        elif not update_dictionary and update_weights:
            _check_init_dictionary(dictionary, (self.n_components, n_samples),
                                   '_aa (input dictionary)')
            weights = _initialize_kernel_aa_weights(
                kernel, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)
        elif update_dictionary and not update_weights:
            _check_init_weights(weights, (n_samples, self.n_components),
                                '_aa (input weights)')
            dictionary = _initialize_kernel_aa_dictionary(
                kernel, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)
        else:
            dictionary, weights = _initialize_kernel_aa(
                kernel, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)

        if alpha is None:
            alpha = _initialize_kernel_aa_scale_factors_random(
                self.n_components, delta=self.delta,
                random_state=self.random_state)

        else:
            _check_init_scale_factors(alpha, self.delta, (self.n_components,),
                                      '_aa (input scale factors)')

        self.weights = weights.copy()
        self.dictionary = dictionary.copy()
        self.alpha = alpha.copy()

        if data.dtype != self.weights.dtype:
            data = data.astype(self.weights.dtype)

        self.weights, self.dictionary, self.alpha, cost, n_iter, avg_time_per_iter, cost_deltas = \
            _iterate_aa(
                data, self.weights, self.dictionary, self.alpha, delta=self.delta,
                update_weights=update_weights,
                update_dictionary=update_dictionary,
                update_scale_factors=update_scale_factors,
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
                require_monotonic_cost_decrease=self.require_monotonic_cost_decrease,
                stopping_criterion=self.stopping_criterion,
                weights_solver_kwargs=self.weights_solver_kwargs,
                dictionary_solver_kwargs=self.dictionary_solver_kwargs,
                scale_factors_solver_kwargs=self.scale_factors_solver_kwargs)

        if n_iter == self.max_iterations and self.tolerance > 0:
            warnings.warn('Maximum number of iterations %d reached.' %
                          self.max_iterations, UserWarning)

        return cost, n_iter, avg_time_per_iter, cost_deltas

    def fit_transform(self, data, dictionary=None, weights=None, alpha=None,
                      **kwargs):
        """Perform archetypal analysis and return transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        data : array-like, shape (n_features, n_samples)
            Data matrix to be factorized.

        dictionary : array-like, shape (n_components, n_samples)
            If init='custom', used as initial guess for solution.

        weights : array-like, shape (n_samples, n_components)
            If init='custom', used as initial guess for solution.

        alpha : array-like, shape (n_components,)
            If init='custom', used as initial guess for solution.

        Returns
        -------
        weights : array-like, shape (n_samples, n_components)
            Representation of the data.
        """

        cost_, n_iter_, avg_time_per_iter_, cost_deltas_ = self._aa(
            data,
            dictionary=dictionary,
            weights=weights, alpha=alpha, **kwargs)

        self.cost = cost_

        if self.delta != 0:
            self.dictionary = np.dot(np.diag(self.alpha), self.dictionary)

        self.archetypes = self.dictionary.dot(data)
        self.n_iter = n_iter_
        self.avg_time_per_iter = avg_time_per_iter_
        self.cost_deltas = cost_deltas_

        return self.weights

    def transform(self, data):
        """Transform the data according to the fitted factorization.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Data matrix to be transformed.

        Returns
        -------
        weights : array-like, shape (n_samples, n_components)
            Representation of the data.

        cost : float
            Cost for calculated representation.
        """

        n_samples = data.shape[0]

        gamma = self.weights_solver_kwargs.get('gamma', 1e-4)
        memory = self.weights_solver_kwargs.get('memory', 1)
        sigma_one = self.weights_solver_kwargs.get('sigma_one', 0.1)
        sigma_two = self.weights_solver_kwargs.get('sigma_two', 0.9)
        lambda_min = self.weights_solver_kwargs.get('lambda_min', 1e-10)
        alpha0 = self.weights_solver_kwargs.get('alpha0', -1.0)
        alpha_min = self.weights_solver_kwargs.get('alpha_min', 1e-5)
        alpha_max = self.weights_solver_kwargs.get('alpha_max', 1e3)
        epsilon_one = self.weights_solver_kwargs.get('epsilon_one', 1e-10)
        epsilon_two = self.weights_solver_kwargs.get('epsilon_two', 1e-6)
        max_feval = self.weights_solver_kwargs.get('max_feval', 2000)

        CKCt = self.archetypes.dot(self.archetypes.T)
        CK = self.archetypes.dot(data.T)

        initial_weights = right_stochastic_matrix(
            (n_samples, self.n_components), random_state=self.random_state)

        # pylint: disable=no-value-for-parameter
        self.weights = _gu_update_kernel_aa_weights(
            CKCt, CK, initial_weights, gamma, memory,
            sigma_one, sigma_two, lambda_min,
            alpha0, alpha_min, alpha_max,
            epsilon_one, epsilon_two,
            self.max_iterations, max_feval)

        cost = (0.5 * np.linalg.norm(
            data - self.weights.dot(self.archetypes)) ** 2 / n_samples)

        return self.weights, cost

    def inverse_transform(self, weights):
        """Transform data back into its original space.

        Parameters
        ----------
        weights : array-like, shape (n_samples, n_components)
            Representation of the data matrix.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_features)
            Weights transformed to original space.
        """

        return weights.dot(self.archetypes)
