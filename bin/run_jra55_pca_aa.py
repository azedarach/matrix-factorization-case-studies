"""
Run archetypal analysis on PCs of JRA-55 500 hPa height anomalies.
"""

# License: MIT

from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
import time

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
import xarray as xr

from convex_dim_red import ArchetypalAnalysis


TIME_NAME = 'initial_time0_hours'

# First and last years to retain for analysis
START_YEAR = 1870
END_YEAR = 2018

# Number of random restarts to use
INIT = 'random'
N_INIT = 100

# Stopping criteria
MAX_ITERATIONS = 10000
TOLERANCE = 1e-6


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Run archetypal analysis on PCs of JRA-55 500 hPa height anomalies'
    )

    parser.add_argument('input_file',
                        help='input file containing SST anomalies')
    parser.add_argument('output_file',
                        help='name of output file')
    parser.add_argument('--n-components', dest='n_components', type=int,
                        default=1, help='number of archetypes')
    parser.add_argument('--delta', dest='delta', type=float,
                        default=0.0, help='dictionary regularization')
    parser.add_argument('--init', dest='init', choices=['random', 'furthest_sum'],
                        default=INIT, help='initialization method')
    parser.add_argument('--n-init', dest='n_init', type=int,
                        default=N_INIT, help='number of initializations')
    parser.add_argument('--tolerance', dest='tolerance', type=float,
                        default=TOLERANCE, help='stopping tolerance')
    parser.add_argument('--max-iterations', dest='max_iterations', type=int,
                        default=MAX_ITERATIONS, help='maximum number of iterations')
    parser.add_argument('--random-seed', dest='random_seed', type=int,
                        default=None, help='random seed')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='produce verbose output')
    parser.add_argument('--restrict-to-base-period',
                        dest='restrict_to_base_period',
                        action='store_true',
                        help='perform analysis only over base period')
    parser.add_argument('--standardize', dest='standardize', action='store_true',
                        help='standardize features before clustering')

    args = parser.parse_args()

    if args.n_components < 1:
        raise ValueError('Number of clusters must be at least 1')

    if args.delta < 0:
        raise ValueError('Relaxation parameter must be non-negative')

    if args.tolerance < 0:
        raise ValueError('Stopping tolerance must be positive')

    if args.n_init < 1:
        raise ValueError('Number of initializations must be at least 1')

    if args.max_iterations < 1:
        raise ValueError('Maximum number of iterations must be at least 1')

    return args


def weight_and_flatten_data(da, weights=None, sample_dim=TIME_NAME):
    """Apply weighting to data and convert to 2D array."""

    feature_dims = [d for d in da.dims if d != sample_dim]
    original_shape = [da.sizes[d] for d in da.dims if d != sample_dim]

    if weights is not None:
        weighted_da = (weights * da).transpose(*da.dims)
    else:
        weighted_da = da

    if weighted_da.get_axis_num(sample_dim) != 0:
        weighted_da = weighted_da.transpose(*([sample_dim] + feature_dims))

    n_samples = weighted_da.sizes[sample_dim]
    n_features = np.product(original_shape)

    flat_data = weighted_da.data.reshape(n_samples, n_features)

    return flat_data


def fit_aa_model(X, n_components=2, delta=0, init=INIT, n_init=N_INIT,
                 tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS,
                 verbose=False, random_state=None, **kwargs):
    """Run archetypal analysis on given data."""

    rng = check_random_state(random_state)

    dictionary_solver_kwargs = dict(max_iterations=1)
    weights_solver_kwargs = dict(max_iterations=1)

    min_cost = None
    best_model = None
    for _ in range(n_init):

        model = ArchetypalAnalysis(
            n_components=n_components, delta=delta,
            init=init, tolerance=tolerance,
            max_iterations=max_iterations, verbose=verbose,
            stopping_criterion='rel_delta_f', random_state=rng,
            dictionary_solver_kwargs=dictionary_solver_kwargs,
            weights_solver_kwargs=weights_solver_kwargs,
            **kwargs)

        model.fit_transform(X)

        if min_cost is None or model.cost < min_cost:
            best_model = deepcopy(model)
            min_cost = model.cost

    return best_model


def run_aa(da, n_components=2, delta=0, standardize=False,
           init=INIT, n_init=N_INIT,
           max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE,
           verbose=False, random_state=None,
           sample_dim=TIME_NAME):
    """Run archetypal analysis on PCs data."""

    rng = check_random_state(random_state)

    feature_dims = [d for d in da.dims if d != sample_dim]
    original_shape = [da.sizes[d] for d in da.dims if d != sample_dim]

    # Convert input data array to plain 2D array
    flat_data = weight_and_flatten_data(da, sample_dim=sample_dim)

    n_features = flat_data.shape[1]

    # Remove any features/columns with missing data
    missing_features = np.any(np.isnan(flat_data), axis=0)
    valid_data = flat_data[:, np.logical_not(missing_features)]

    if standardize:
        valid_data = valid_data / np.std(valid_data, axis=0, keepdims=True)

    start_time = time.perf_counter()
    best_model = fit_aa_model(
        valid_data, n_components=n_components, delta=delta,
        init=init, n_init=n_init,
        tolerance=tolerance, max_iterations=max_iterations, verbose=verbose,
        random_state=rng)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    reconstruction = best_model.inverse_transform(best_model.weights)

    cost = best_model.cost
    rmse = mean_squared_error(
        valid_data, reconstruction, squared=False)

    aa_weights_da = xr.DataArray(
        best_model.weights, coords={sample_dim: da[sample_dim],
                                    'archetype': np.arange(n_components)},
        dims=[sample_dim, 'archetype'])

    aa_dictionary_da = xr.DataArray(
        best_model.dictionary,
        coords={'archetype': np.arange(n_components),
                sample_dim: da[sample_dim]},
        dims=['archetype', sample_dim])

    valid_archetypes = best_model.archetypes
    full_archetypes = np.full((n_components, n_features), np.NaN)
    full_archetypes[:, np.logical_not(missing_features)] = valid_archetypes
    archetypes = np.reshape(
        full_archetypes, [n_components,] + original_shape)

    archetype_coords = {d: da[d] for d in feature_dims}
    archetype_coords['archetype'] = np.arange(n_components)

    archetype_dims = ['archetype'] + feature_dims

    aa_archetypes_da = xr.DataArray(
        archetypes, coords=archetype_coords, dims=archetype_dims)

    cost_deltas_da = xr.DataArray(
        best_model.cost_deltas,
        coords={'iteration': np.arange(best_model.n_iter + 1)},
        dims=['iteration'])

    data_vars = {'weights': aa_weights_da,
                 'dictionary': aa_dictionary_da,
                 'archetypes': aa_archetypes_da,
                 'cost_deltas': cost_deltas_da}

    aa_ds = xr.Dataset(data_vars)

    aa_ds.attrs['cost'] = '{:16.8e}'.format(cost)
    aa_ds.attrs['rmse'] = '{:16.8e}'.format(rmse)
    aa_ds.attrs['n_iter'] = '{:d}'.format(best_model.n_iter)
    aa_ds.attrs['avg_time_per_iter'] = '{:16.8e}'.format(
        best_model.avg_time_per_iter)

    aa_ds.attrs['init'] = init
    aa_ds.attrs['n_init'] = '{:d}'.format(n_init)
    aa_ds.attrs['delta'] = '{:16.8e}'.format(delta)
    aa_ds.attrs['max_iterations'] = '{:d}'.format(max_iterations)
    aa_ds.attrs['tolerance'] = '{:16.8e}'.format(tolerance)
    aa_ds.attrs['elapsed_time'] = '{:16.8e}'.format(elapsed_time)

    return aa_ds


def main():
    """Run archetypal analysis on PCs of JRA-55 500 hPa height anomalies."""

    args = parse_cmd_line_args()

    random_state = check_random_state(args.random_seed)

    var_name = 'PCs'

    with xr.open_dataset(args.input_file) as hgt_eofs_ds:

        clim_base_period = [int(hgt_eofs_ds.attrs['eofs_start_year']),
                            int(hgt_eofs_ds.attrs['eofs_end_year'])]

        hgt_pcs_da = hgt_eofs_ds[var_name].where(
            (hgt_eofs_ds[TIME_NAME].dt.year >= START_YEAR) &
            (hgt_eofs_ds[TIME_NAME].dt.year <= END_YEAR), drop=True)


        if args.restrict_to_base_period:
            hgt_pcs_da = hgt_pcs_da.where(
                (hgt_pcs_da[TIME_NAME].dt.year >= clim_base_period[0]) &
                (hgt_pcs_da[TIME_NAME].dt.year <= clim_base_period[1]), drop=True)

        aa_ds = run_aa(
            hgt_pcs_da, n_components=args.n_components, delta=args.delta,
            standardize=args.standardize, init=args.init, n_init=args.n_init,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance, verbose=args.verbose,
            random_state=random_state)

        aa_ds.attrs['input_file'] = args.input_file
        aa_ds.attrs['base_period_start_year'] = '{:d}'.format(clim_base_period[0])
        aa_ds.attrs['base_period_end_year'] = '{:d}'.format(clim_base_period[1])
        aa_ds.attrs['random_seed'] = '{:d}'.format(args.random_seed)

        aa_ds.to_netcdf(args.output_file)


if __name__ == '__main__':
    main()
