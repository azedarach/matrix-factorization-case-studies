"""
Provides routines for performing spectral projected gradient descent.
"""

# License: MIT

from __future__ import absolute_import, division, print_function


import time
import warnings

import numpy as np
from numba import jit

from .simplex_projection import simplex_project_vector


@jit(nopython=True)
def spg_line_search_step_length(current_step_length, delta, f_old, f_new,
                                sigma_one=0.1, sigma_two=0.9):
    """Return next step length for line search."""

    step_length_tmp = (-0.5 * current_step_length ** 2 * delta /
                       (f_new - f_old - current_step_length * delta))

    next_step_length = 0
    if sigma_one <= step_length_tmp <= sigma_two * current_step_length:
        next_step_length = step_length_tmp
    else:
        next_step_length = 0.5 * current_step_length

    return next_step_length


@jit(nopython=True)
def spg_line_search_cauchy_step_size(beta, sksk, alpha_min=1e-3, alpha_max=1e3):
    """Return next value of Cauchy step size parameter in SPG optimization."""

    if beta <= 0:
        return alpha_max

    return min(alpha_max, max(alpha_min, sksk / beta))


def spg(f, df, x0, project=None, gamma=1e-4, memory=1,
        sigma_one=0.1, sigma_two=0.9, lambda_min=1e-10,
        alpha0=None, alpha_min=1e-5, alpha_max=1e3,
        epsilon_one=1e-10, epsilon_two=1e-6,
        use_infinity_norm=True, verbose=0,
        max_iterations=10000, max_feval=1000000):
    """Perform gradient descents steps with non-monotone line-search.

    Parameters
    ----------
    f : callable
        Object callable with signature f(x), returning a scalar value.

    df : callable
        Object callable with signature df(x), returning a vector with the
        same shape as x containing the gradient of the function f.

    x0 : array-like
        Initial guess for the solution.

    project : optional
        If given, must be callable with signature project(x), returning a
        vector with the same shape as x containing the projection of x
        into the feasible region.

    gamma : float, default: 1e-4
        Sufficient decrease parameter for line-search.

    memory : integer, default: 1
        Number of previous function values to use in non-monotone line-search.

    sigma_one : float, default: 0.1
        Lower safe-guarding parameter for acceptable step-sizes.

    sigma_two : float, default: 0.9
        Upper safe-guarding parameter for acceptable step-sizes.

    lambda_min : float, default: 1e-10
        Minimum allowed line-search step-size.

    alpha0 : float, optional
        If given, initial value for descent step-size.

    alpha_min : float, default: 1e-5
        Minimum allowed value for descent step-size.

    alpha_max : float, default: 1e3
        Maximum allowed value for descent step-size.

    epsilon_one : float, default: 1e-10
        Stopping tolerance used for infinity norm convergence criterion.

    epsilon_two : float, default: 1e-6
        Stopping tolerance used for 2-norm convergence criterion.

    use_infinity_norm : bool, default: True
        If True, include condition on infinity norm of increment in
        convergence check. If False, only use 2-norm convergence criterion.

    verbose : integer, default: 0
        If True, produce verbose output.

    max_iterations : integer
        Maximum allowed number of iterations.

    max_feval : integer
        Maximum allowed number of function evaluations.

    Returns
    -------
    sol : array-like
        Estimate for the minimum of f.

    fmin : float
        Value of the function at the found solution.

    n_iter : integer
        Number of descent steps required.

    n_feval : integer
        Number of function evaluations required.

    References
    ----------
    E. G. Birgin, J. M. Martinez, and M. Raydan, "Algorithm 813:
    SPG—Software for Convex-Constrained Optimization",
    ACM Trans. Math. Softw. 27, 3 (2001), 340-349,
    doi:10.1145/502800.502803.
    """

    is_multivariate = not np.isscalar(x0)

    # Pre-allocate arrays for gradients and increments
    x = x0.copy() if is_multivariate else x0
    x_old = np.zeros_like(x)
    gk = np.zeros_like(x)
    dk = np.zeros_like(x)
    yk = np.zeros_like(x)
    res = np.zeros_like(x)

    # Ensure initial guess is in the feasible region.
    if project is not None:
        x = project(x)

    # Initialize line search parameters
    alpha = alpha0

    f_mem = np.zeros(memory)

    # Calculate current value of cost function
    f_old = f(x)
    n_feval = 1

    if verbose:
        print('{:<12s} | {:<12s} | {:<13s} | {:<13s} | {:<12s}'.format(
            'n_iter', 'n_feval', 'f', 'conv_crit', 'time'))
        print('-' * 79)
        print('{:12d} | {:12d} | {: 12.6e} | {: 12.6e} | {: 12.6e}'.format(
            0, n_feval, f_old, -1, 0))

    # Update solution by performing gradient descent steps.
    has_converged = False
    for n_iter in range(max_iterations):

        start_time = time.perf_counter()

        # Save current estimate for solution.
        x_old = x.copy() if is_multivariate else x

        # Compute gradient and current increment.
        gk = df(x)

        if alpha is None:
            # If on the first iteration and no initial value of alpha
            # given, use the inverse of the infinity norm of the increment.
            if project is None:
                alpha = 1.0 / np.max(np.abs(gk))
            else:
                alpha_inv = np.max(np.abs(project(x - gk) - x))

                if abs(alpha_inv) > 1e-12:
                    alpha = 1.0 / alpha_inv
                else:
                    alpha = 1.0

        dk = -alpha * gk   # pylint: disable=invalid-unary-operand-type
        if project is not None:
            dk = project(x + dk)
            dk -= x

        # Determine target function value for line search.
        f_mem = np.roll(f_mem, 1)
        f_mem[0] = f_old

        f_max = None
        for previous_value in f_mem:
            if f_max is None or previous_value >= f_max:
                f_max = previous_value

        # Update step-size by performing line search.
        delta = np.sum(dk * gk)
        lam = 1

        x = x_old + dk

        f_new = f(x)
        n_feval += 1

        while f_new > f_max + gamma * lam * delta:

            lam = spg_line_search_step_length(
                lam, delta, f_old, f_new,
                sigma_one=sigma_one, sigma_two=sigma_two)

            x = x_old + lam * dk

            f_new = f(x)
            n_feval += 1

            if abs(lam) < lambda_min:
                warnings.warn(
                    'step size below tolerance in SPG line search',
                    UserWarning)
                break

        # Determine change in gradient vector between old and new estimate.
        yk = gk.copy() if is_multivariate else gk
        gk = df(x)
        yk = gk - yk

        # Update line search step size
        sksk = lam ** 2 * np.sum(dk * dk)
        betak = lam * np.sum(dk * yk)

        alpha = spg_line_search_cauchy_step_size(
            betak, sksk, alpha_min=alpha_min, alpha_max=alpha_max)

        f_old = f(x)
        n_feval += 1

        # Check for convergence of solution.
        if project is None:
            res = -gk
        else:
            res = project(x - gk) - x

        res_norm = np.sum(res ** 2) ** 0.5

        end_time = time.perf_counter()

        if verbose:
            print('{:12d} | {:12d} | {: 12.6e} | {: 12.6e} | {: 12.6e}'.format(
                n_iter + 1, n_feval, f_old, res_norm,
                end_time - start_time))

        has_converged = res_norm < epsilon_two
        if use_infinity_norm:
            has_converged = has_converged or np.max(np.abs(res)) < epsilon_one

        if has_converged:
            if verbose:
                print('-' * 79)
                print('*** Converged at iteration {:d} ***'.format(n_iter + 1))
            break

        # If number of function evaluations is exceeded, stop iteration.
        if n_feval > max_feval:
            warnings.warn(
                'maximum number of function evaluations exceeded in SPG',
                UserWarning)
            break

    if n_iter == max_iterations - 1 and not has_converged:
        warnings.warn(
            'maximum number of iterations exceeded in SPG',
            UserWarning)

    return x, f_old, n_iter, n_feval


@jit(nopython=True)
def quad_simplex_spg(A, b, x0, gamma=1e-4, memory=1,
                     sigma_one=0.1, sigma_two=0.9, lambda_min=1e-10,
                     alpha0=-1.0, alpha_min=1e-5, alpha_max=1e3,
                     epsilon_one=1e-10, epsilon_two=1e-6,
                     max_iterations=1000, max_feval=2000):
    """Solve quadratic program constrained to standard simplex.

    Finds a minimizer of f(x) = x.T * A * x / 2 + b * x where
    x is constrained to lie in the standard simplex.
    """

    # Ensure initial guess is feasible.
    x = simplex_project_vector(x0)

    x_old = np.zeros_like(x)

    Ax = np.zeros_like(x)
    gk = np.zeros_like(x)
    dk = np.zeros_like(x)
    yk = np.zeros_like(x)
    res = np.zeros_like(x)

    # Initialize line search parameters
    f_mem = np.full(memory, np.NaN)

    # Calculate current value of cost function
    Ax = A.dot(x)
    f_old = 0.5 * x.dot(Ax) + x.dot(b)
    n_feval = 1

    # Update solution by performing gradient descent steps.
    for n_iter in range(max_iterations):

        # Save current estimate for solution.
        x_old[:] = x[:]

        # Compute gradient and current increment.
        gk = Ax + b

        if n_iter == 0:
            # If on the first iteration, initialize descent step-size.
            if alpha_min <= alpha0 <= alpha_max:
                alpha = alpha0
            else:
                alpha_inv = np.max(np.abs(simplex_project_vector(x - gk) - x))

                if abs(alpha_inv) < 1e-12:
                    alpha_inv = 1.0

                alpha = min(max(alpha_min, 1.0 / alpha_inv), alpha_max)

        dk = simplex_project_vector(x - alpha * gk) - x

        # Determine target function value for line search.
        f_mem = np.roll(f_mem, 1)
        f_mem[0] = f_old

        f_max = np.nanmax(f_mem)

        # Update step-size by performing line search.
        delta = dk.dot(gk)
        lam = 1

        x = x_old + dk
        Ax = A.dot(x)

        f_new = 0.5 * x.dot(Ax) + x.dot(b)
        n_feval += 1

        while f_new > f_max + gamma * lam * delta:

            lam = spg_line_search_step_length(
                lam, delta, f_old, f_new,
                sigma_one=sigma_one, sigma_two=sigma_two)

            x = x_old + lam * dk
            Ax = A.dot(x)
            f_new = 0.5 * x.dot(Ax) + x.dot(b)
            n_feval += 1

            if abs(lam) < lambda_min:
                break

        # Determine change in gradient vector between old and new estimate.
        yk = Ax + b - gk
        gk = yk + gk

        # Update line search step size
        sksk = lam ** 2 * dk.dot(dk)
        betak = lam * dk.dot(yk)

        alpha = spg_line_search_cauchy_step_size(
            betak, sksk, alpha_min=alpha_min, alpha_max=alpha_max)

        f_old = 0.5 * x.dot(Ax) + x.dot(b)
        n_feval += 1

        # Check for convergence of solution.
        res = simplex_project_vector(x - gk) - x

        res_norm = np.sum(res ** 2) ** 0.5

        has_converged = res_norm < epsilon_two or np.max(np.abs(res)) < epsilon_one

        if has_converged:
            break

        # If number of function evaluations is exceeded, stop iteration.
        if n_feval > max_feval:
            break

    return x
