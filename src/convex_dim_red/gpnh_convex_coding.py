"""
Provides routines for performing GPNH-regularized convex coding.
"""

# License: MIT

from __future__ import absolute_import, division, print_function

import numbers
import time
import warnings

import numpy as np
from sklearn.utils import check_array, check_random_state
from numba import guvectorize, jit, float64, int32

from .furthest_sum import furthest_sum
from .spg import quad_simplex_spg
from .stochastic_matrices import right_stochastic_matrix
from .validation_utils import check_unit_axis_sums, check_array_shape


INTEGER_TYPES = (numbers.Integral, np.integer)

INITIALIZATION_METHODS = (None, 'random', 'furthest_sum',)


def _check_init_weights(weights, shape, whom):

    weights = check_array(weights)
    check_array_shape(weights, shape, whom)
    check_unit_axis_sums(weights, whom, axis=1)


def _check_init_dictionary(dictionary, shape, whom):

    dictionary = check_array(dictionary)
    check_array_shape(dictionary, shape, whom)


def _initialize_gpnh_convex_coding_dictionary_random(data, n_components,
                                                     random_state=None):
    rng = check_random_state(random_state)

    n_features = data.shape[1]
    avg = np.sqrt(np.abs(data).mean() / n_components)
    dictionary = avg * rng.randn(n_features, n_components)

    return dictionary


def _initialize_gpnh_convex_coding_dictionary_furthest_sum(
        data, n_components, start_index=None, n_extra_steps=10,
        exclude=None, random_state=None):
    rng = check_random_state(random_state)

    n_features = data.shape[1]
    kernel = data.dot(data.T)

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

    dictionary = np.zeros((n_features, n_components),
                          dtype=kernel.dtype)
    for i in range(n_components):
        dictionary[:, i] = data[selected[i]]

    return dictionary


def _initialize_gpnh_convex_coding_weights_random(data, n_components,
                                                  random_state=None):
    rng = check_random_state(random_state)

    n_samples = data.shape[0]

    return right_stochastic_matrix((n_samples, n_components), random_state=rng)


def _initialize_gpnh_convex_coding_dictionary(data, n_components, init='random',
                                              random_state=None, **kwargs):
    if init is None:
        init = 'random'

    if init == 'random':
        return _initialize_gpnh_convex_coding_dictionary_random(
            data, n_components, random_state=random_state)

    if init == 'furthest_sum':
        start_index = kwargs.get('start_index', None)
        n_extra_steps = kwargs.get('n_extra_steps', 10)
        exclude = kwargs.get('exclude', None)

        return _initialize_gpnh_convex_coding_dictionary_furthest_sum(
            data, n_components, start_index=start_index,
            n_extra_steps=n_extra_steps, exclude=exclude,
            random_state=random_state)

    raise ValueError(
        'Invalid init parameter: got %r instead of one of %r' %
        (init, INITIALIZATION_METHODS))


def _initialize_gpnh_convex_coding_weights(data, n_components, init='random',
                                           random_state=None):
    if init is None:
        init = 'random'

    if init in ('furthest_sum', 'random'):
        return _initialize_gpnh_convex_coding_weights_random(
            data, n_components, random_state=random_state)

    raise ValueError(
        'Invalid init parameter: got %r instead of one of %r' %
        (init, INITIALIZATION_METHODS))


def _initialize_gpnh_convex_coding(data, n_components, init='random',
                                   random_state=None, **kwargs):
    if init is None:
        init = 'random'

    rng = check_random_state(random_state)

    dictionary = _initialize_gpnh_convex_coding_dictionary(
        data, n_components, init=init, random_state=rng, **kwargs)
    weights = _initialize_gpnh_convex_coding_weights(
        data, n_components, init=init, random_state=rng)

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
def _gpnh_regularization(dictionary):
    """Evaluate GPNH regularization term."""

    n_features, n_components = dictionary.shape

    phi_W = 0.0

    if n_components == 1:
        return phi_W

    prefactor = 2.0 / (n_components * n_features * (n_components - 1.0))

    for i in range(n_components):
        for j in range(i + 1, n_components):
            phi_W += np.linalg.norm(dictionary[:, i] - dictionary[:, j]) ** 2

    return prefactor * phi_W


@jit(nopython=True)
def _gpnh_cost(data, weights, dictionary, lambda_W=0):
    """Evaluate GPNH convex coding cost function."""

    n_samples = data.shape[0]

    cost = 0.5 * np.linalg.norm(data - weights.dot(dictionary.T)) ** 2 / n_samples

    if lambda_W != 0:
        cost += lambda_W * _gpnh_regularization(dictionary)

    return cost


def _update_gpnh_dictionary(X, weights, ZtZ, GW, lambda_W=0):
    """Update dictionary for GPNH regularized convex coding."""

    n_samples = X.shape[0]

    # Compute gradient terms.
    ZtX = weights.T.dot(X)

    lhs = ZtZ / n_samples + lambda_W * GW
    rhs = ZtX / n_samples

    sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return sol.T


@guvectorize(
    [(float64[:, :], float64[:, :], float64[:, :],
      float64, int32, float64, float64, float64, float64, float64, float64,
      float64, float64, int32, int32, float64[:, :])],
    '(k, k), (i, k), (i, k), (), (), (), (), (), (), (), (), (), (), (), () -> (i, k)',
    nopython=True, target='parallel')
def _gu_update_gpnh_weights(WtW, XW, initial_weights,
                            gamma, memory,
                            sigma_one, sigma_two, lambda_min,
                            alpha0, alpha_min, alpha_max,
                            epsilon_one, epsilon_two,
                            max_iterations, max_feval, final_weights):

    n_samples = initial_weights.shape[0]

    for t in range(n_samples):

        final_weights[t] = quad_simplex_spg(
            WtW, -XW[t], initial_weights[t], gamma=gamma, memory=memory,
            sigma_one=sigma_one, sigma_two=sigma_two, lambda_min=lambda_min,
            alpha0=alpha0, alpha_min=alpha_min, alpha_max=alpha_max,
            epsilon_one=epsilon_one, epsilon_two=epsilon_two,
            max_iterations=max_iterations, max_feval=max_feval)


def _update_gpnh_weights(X, weights, dictionary, **solver_kwargs):
    """Update weights for GPNH regularized convex coding."""

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

    WtW = dictionary.T.dot(dictionary)
    XW = X.dot(dictionary)

    # pylint: disable=no-value-for-parameter
    return _gu_update_gpnh_weights(
        WtW, XW, weights, gamma, memory,
        sigma_one, sigma_two, lambda_min,
        alpha0, alpha_min, alpha_max,
        epsilon_one, epsilon_two,
        max_iterations, max_feval)


def _iterate_gpnh_convex_coding(X, weights, dictionary, lambda_W=0,
                                update_weights=True, update_dictionary=True,
                                tolerance=1e-6, max_iterations=1000, verbose=0,
                                **kwargs):
    """Iteratively update weights and dictionary until convergence is reached."""

    n_features = X.shape[1]
    n_samples, n_components = weights.shape

    # Pre-compute constants.
    WtXt = dictionary.T.dot(X.T)
    ZtZ = weights.T.dot(weights)
    WtW = dictionary.T.dot(dictionary)

    if n_components > 1:
        prefactor = (4.0 / (n_features * n_components * (n_components - 1)))
        GW = prefactor * (n_components * np.eye(n_components) - 1)
    else:
        GW = np.zeros((n_components, n_components))

    trace_XtX = X.T.dot(X).trace()
    trace_WtXtZ = WtXt.dot(weights).trace()
    trace_ZtZWtW = ZtZ.dot(WtW).trace()
    dictionary_penalty = 0
    if lambda_W != 0:
        dictionary_penalty = lambda_W * _gpnh_regularization(dictionary)

    new_cost = (0.5 * (trace_XtX - 2 * trace_WtXtZ + trace_ZtZWtW) /
                n_samples + dictionary_penalty)

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

    # Iterate until stopping criteria are satisfied or maximum number of
    # iterations is reached.
    iter_times = []
    cost_deltas = []

    if verbose:
        print("*** GPNH convex coding: n_components = {:d} ***".format(
            n_components))
        print('{:<12s} | {:<13s} | {:<13s} | {:<12s}'.format(
            'Iteration', 'Cost', 'Cost delta', 'Time'))
        print(100 * '-')

    for n_iter in range(max_iterations):

        start_time = time.perf_counter()

        old_cost = new_cost

        if update_dictionary:

            # Find optimal dictionary for fixed weights.
            dictionary = _update_gpnh_dictionary(
                X, weights, ZtZ, GW, lambda_W=lambda_W,
                **dictionary_solver_kwargs)

            WtXt = dictionary.T.dot(X.T)
            WtW = dictionary.T.dot(dictionary)

            trace_WtXtZ = WtXt.dot(weights).trace()
            trace_ZtZWtW = ZtZ.dot(WtW).trace()
            dictionary_penalty = 0
            if lambda_W != 0:
                dictionary_penalty = lambda_W * _gpnh_regularization(dictionary)

            new_cost = (0.5 * (trace_XtX - 2 * trace_WtXtZ + trace_ZtZWtW) /
                        n_samples + dictionary_penalty)

            _check_if_cost_increased(
                old_cost, new_cost, tolerance, 'dictionary',
                require_decrease=require_monotonic_cost_decrease)

        if update_weights:

            # Find optimal weights for fixed dictionary.
            weights = _update_gpnh_weights(
                X, weights, dictionary, **weights_solver_kwargs)

            ZtZ = weights.T.dot(weights)

            trace_WtXtZ = WtXt.dot(weights).trace()
            trace_ZtZWtW = ZtZ.dot(WtW).trace()

            new_cost = (0.5 * (trace_XtX - 2 * trace_WtXtZ + trace_ZtZWtW) /
                        n_samples + dictionary_penalty)

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

    return (weights, dictionary, new_cost, n_iter,
            np.mean(iter_times), cost_deltas)


class GPNHConvexCoding():
    """Convex encoding of data with GPNH regularization.

    Parameters
    ----------
    n_components : integer or None
        If an integer, the number of components. If None, then all
        features are kept.

    lambda_W : float, default: 0
        Regularization parameter for the dictionary.

    init : None | 'furthest_sum' | 'random' | 'custom'
        Method used to initialize the algorithm.
        Default: None
        Valid options:

        - None: falls back to 'random'.

        - 'furthest_sum': dictionary is initialized using FurthestSum
          method, and weights are initialized to a random stochastic
          matrix.

        - 'random': random matrix of dictionary elements scaled by
          sqrt(X.mean() / n_components), and a random stochastic
          matrix of weights.

        - 'custom': use custom matrices for dictionary and weights.

    tolerance : float, default: 1e-6
        Tolerance of the stopping condition.

    max_iterations : integer, default: 1000
        Maximum number of iterations before stopping.

    verbose : integer, default: 0
        The verbosity level.

    random_state : integer, RandomState, or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    Attributes
    ----------
    dictionary_ : array-like, shape (n_features, n_components)
        The dictionary of states.

    cost_ : number
        Value of the cost function for the obtained factorization.

    n_iter_ : integer
        Actual number of iterations.

    Examples
    --------
    import numpy as np
    X = np.random.rand(4, 10)
    from gpnh_convex_coding import GPNHConvexCoding
    model = GPNHConvexCoding(n_components=2, init='random', random_state=0)
    weights = model.fit_transform(X)
    dictionary = model.dictionary_

    References
    ----------
    S. Gerber, L. Pospisil, M. Navandard, and I. Horenko,
    "Low-cost scalable discretization, prediction and feature selection
    for complex systems" (2018)
    """
    def __init__(self, n_components,
                 lambda_W=0, init=None,
                 tolerance=1e-6, max_iterations=1000,
                 verbose=0, random_state=None, **kwargs):
        self.n_components = n_components
        self.lambda_W = lambda_W
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
        self.cost = 0
        self.n_iter = 0
        self.avg_time_per_iter = 0
        self.cost_deltas = None

        self.weights_solver_kwargs = kwargs.get('weights_solver_kwargs', {})
        self.dictionary_solver_kwargs = kwargs.get('dictionary_solver_kwargs', {})

    def _gpnh_convex_coding(self, data, dictionary=None, weights=None,
                            update_dictionary=True, update_weights=True,
                            **kwargs):
        """Calculate GPNH-regularized convex coding of dataset."""

        n_samples, n_features = data.shape

        if self.n_components is None:
            self.n_components = n_features

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
                                '_gpnh_convex_coding (input weights)')
            _check_init_dictionary(dictionary, (n_features, self.n_components),
                                   '_gpnh_convex_coding (input dictionary)')

        elif not update_dictionary and update_weights:

            _check_init_dictionary(dictionary, (n_features, self.n_components),
                                   '_gpnh_convex_coding (input dictionary)')
            weights = _initialize_gpnh_convex_coding_weights(
                data, self.n_components, init=self.init,
                random_state=self.random_state)

        elif update_dictionary and not update_weights:

            _check_init_weights(weights, (n_samples, self.n_components),
                                '_gpnh_convex_coding (input weights)')
            dictionary = _initialize_gpnh_convex_coding_dictionary(
                data, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)

        else:

            dictionary, weights = _initialize_gpnh_convex_coding(
                data, self.n_components, init=self.init,
                random_state=self.random_state, **kwargs)

        self.weights = weights.copy()
        self.dictionary = dictionary.copy()

        self.weights, self.dictionary, cost, n_iter, avg_time_per_iter, cost_deltas = \
            _iterate_gpnh_convex_coding(
                data, self.weights, self.dictionary, lambda_W=self.lambda_W,
                update_dictionary=update_dictionary,
                update_weights=update_weights,
                tolerance=self.tolerance,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
                require_monotonic_cost_decrease=self.require_monotonic_cost_decrease,
                stopping_criterion=self.stopping_criterion,
                weights_solver_kwargs=self.weights_solver_kwargs,
                dictionary_solver_kwargs=self.dictionary_solver_kwargs)

        if n_iter == self.max_iterations and self.tolerance > 0:
            warnings.warn('Maximum number of iterations %d reached.' %
                          self.max_iterations, UserWarning)

        return cost, n_iter, avg_time_per_iter, cost_deltas

    def fit_transform(self, data, dictionary=None, weights=None, **kwargs):
        """Fit convex coding and return transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Data matrix to be factorized.

        dictionary : array-like, shape (n_features, n_components)
            If init='custom', used as initial guess for solution.

        weights : array-like, shape (n_samples, n_components)
            If init='custom', used as initial guess for solution.

        Returns
        -------
        weights : array-like, shape (n_samples, n_components)
            Representation of the data.
        """

        cost_, n_iter_, avg_time_per_iter_, cost_deltas_ = self._gpnh_convex_coding(
            data,
            dictionary=dictionary,
            weights=weights, **kwargs)

        self.cost = cost_
        self.n_iter = n_iter_
        self.avg_time_per_iter = avg_time_per_iter_
        self.cost_deltas = cost_deltas_

        return self.weights

    def fit(self, data, **kwargs):
        """Fit convex coding to data.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Data matrix to perform analysis on.

        Returns
        -------
        self
        """
        self.fit_transform(data, **kwargs)
        return self

    def transform(self, data):
        """Transform the data according to the fitted factorization.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Data matrix for data to be transformed.

        Returns
        -------
        weights : array-like, shape (n_samples, n_components)
            Representation of the data.

        cost : float
            Cost associated with the calculated representation.
        """

        cost_ = self._gpnh_convex_coding(
            data=data,
            dictionary=self.dictionary,
            update_dictionary=False, update_weights=True,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            require_monotonic_cost_decrease=self.require_monotonic_cost_decrease,
            stopping_criterion=self.stopping_criterion,
            weights_solver_kwargs=self.weights_solver_kwargs,
            dictionary_solver_kwargs=self.dictionary_solver_kwargs)[0]

        return self.weights, cost_

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

        return weights.dot(self.dictionary.T)
