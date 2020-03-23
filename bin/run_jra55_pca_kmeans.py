"""
Run k-means on PCs of JRA-55 500 hPa height anomalies.
"""

# License: MIT

from __future__ import absolute_import, division, print_function

import argparse
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
import xarray as xr

from convex_dim_red import gap_statistic


TIME_NAME = 'initial_time0_hours'

# First and last years to retain for analysis
START_YEAR = 1958
END_YEAR = 2018


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Run k-means on PCs of JRA-55 500 hPa height anomalies')

    parser.add_argument('input_file',
                        help='input file containing height anomalies')
    parser.add_argument('output_file',
                        help='name of output file')
    parser.add_argument('--n-components', dest='n_components', type=int,
                        default=1, help='number of clusters')
    parser.add_argument('--init', dest='init', choices=['random', 'k-means++'],
                        default='k-means++', help='initialization method')
    parser.add_argument('--n-init', dest='n_init', type=int,
                        default=10, help='number of initializations')
    parser.add_argument('--tolerance', dest='tolerance', type=float,
                        default=1e-4, help='stopping tolerance')
    parser.add_argument('--max-iterations', dest='max_iterations', type=int,
                        default=10000, help='maximum number of iterations')
    parser.add_argument('--random-seed', dest='random_seed', type=int,
                        default=None, help='random seed')
    parser.add_argument('--n-trials', dest='n_trials', type=int,
                        default=100, help='number of gap statistic trials')
    parser.add_argument('--n-jobs', dest='n_jobs', type=int,
                        default=1, help='number of jobs to use')
    parser.add_argument('--reference', dest='reference',
                        choices=['uniform', 'pca'], default='uniform',
                        help='gap statistic reference distribution')
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

    if args.tolerance < 0:
        raise ValueError('Stopping tolerance must be positive')

    if args.n_init < 1:
        raise ValueError('Number of initializations must be at least 1')

    if args.max_iterations < 1:
        raise ValueError('Maximum number of iterations must be at least 1')

    if args.n_trials < 1:
        raise ValueError('Number of gap statistic trials must be at least 1')

    if args.n_jobs < 1:
        raise ValueError('Number of jobs must be at least 1')

    return args


def fit_kmeans_model(X, n_components=2, init='k-means++', n_init=100,
                     tolerance=1e-4, max_iterations=10000, n_trials=100,
                     verbose=False, reference='uniform', n_jobs=1,
                     random_state=None):
    """Run k-means on given data."""

    rng = check_random_state(random_state)

    model = KMeans(n_clusters=n_components,
                   init=init, n_init=n_init, tol=tolerance,
                   max_iter=max_iterations, verbose=verbose,
                   random_state=rng).fit(X)

    gap, sk = gap_statistic(X, model.inertia_, n_components=n_components,
                            n_trials=n_trials, reference=reference,
                            n_jobs=n_jobs, random_state=rng)

    return model, gap, sk


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


def run_kmeans(da, n_components=2, standardize=False,
               init='k-means++', n_init=100,
               max_iterations=10000, tolerance=1e-4,
               verbose=False, random_state=None, n_trials=100,
               reference='uniform', n_jobs=1,
               sample_dim=TIME_NAME):
    """Run k-means on SST data."""

    rng = check_random_state(random_state)

    feature_dims = [d for d in da.dims if d != sample_dim]
    original_shape = [da.sizes[d] for d in da.dims if d != sample_dim]

    # Convert input data array to plain 2D array
    flat_data = weight_and_flatten_data(da, sample_dim=sample_dim)

    n_samples, n_features = flat_data.shape

    # Remove any features/columns with missing data
    missing_features = np.any(np.isnan(flat_data), axis=0)
    valid_data = flat_data[:, np.logical_not(missing_features)]

    if standardize:
        normalization = np.std(valid_data, axis=0, keepdims=True)
    else:
        normalization = np.ones((1, valid_data.shape[1]))

    valid_data = valid_data / normalization

    start_time = time.perf_counter()
    best_model, gap, sk = fit_kmeans_model(
        valid_data, n_components=n_components,
        init=init, n_init=n_init, tolerance=tolerance,
        max_iterations=max_iterations, n_trials=n_trials,
        verbose=verbose, reference=reference, n_jobs=n_jobs,
        random_state=rng)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    reconstruction = np.empty_like(valid_data)
    for i in range(n_components):
        mask = best_model.labels_ == i
        reconstruction[mask] = best_model.cluster_centers_[i]

    cost = best_model.inertia_
    rmse = mean_squared_error(valid_data, reconstruction, squared=False)

    kmeans_labels = np.zeros((n_samples, n_components))
    for i in range(n_components):
        mask = best_model.labels_ == i
        kmeans_labels[mask, i] = 1

    kmeans_labels_da = xr.DataArray(
        kmeans_labels,
        coords={sample_dim: da[sample_dim],
                'cluster': np.arange(n_components)},
        dims=[sample_dim, 'cluster'])

    valid_dictionary = best_model.cluster_centers_
    if standardize:
        valid_dictionary = valid_dictionary * normalization

    full_dictionary = np.full((n_components, n_features), np.NaN)
    full_dictionary[:, np.logical_not(missing_features)] = valid_dictionary
    dictionary = np.reshape(
        full_dictionary, [n_components,] + original_shape)

    dictionary_coords = {d: da[d] for d in feature_dims}
    dictionary_coords['cluster'] = np.arange(n_components)

    dictionary_dims = ['cluster'] + feature_dims

    kmeans_dictionary_da = xr.DataArray(
        dictionary, coords=dictionary_coords, dims=dictionary_dims)

    data_vars = {'weights': kmeans_labels_da,
                 'dictionary': kmeans_dictionary_da}

    kmeans_ds = xr.Dataset(data_vars)

    kmeans_ds.attrs['cost'] = '{:16.8e}'.format(cost)
    kmeans_ds.attrs['rmse'] = '{:16.8e}'.format(rmse)
    kmeans_ds.attrs['gap_statistic'] = '{:16.8e}'.format(gap)
    kmeans_ds.attrs['gap_sk'] = '{:16.8e}'.format(sk)
    kmeans_ds.attrs['n_iter'] = '{:d}'.format(best_model.n_iter_)

    kmeans_ds.attrs['init'] = init
    kmeans_ds.attrs['n_init'] = '{:d}'.format(n_init)
    kmeans_ds.attrs['max_iterations'] = '{:d}'.format(max_iterations)
    kmeans_ds.attrs['tolerance'] = '{:16.8e}'.format(tolerance)
    kmeans_ds.attrs['reference'] = '{}'.format(reference)
    kmeans_ds.attrs['n_trials'] = '{:d}'.format(n_trials)
    kmeans_ds.attrs['elapsed_time'] = '{:16.8e}'.format(elapsed_time)

    return kmeans_ds


def main():
    """Run k-means on PCs of JRA-55 500 hPa height anomalies."""

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

        kmeans_ds = run_kmeans(
            hgt_pcs_da, n_components=args.n_components,
            standardize=args.standardize, init=args.init, n_init=args.n_init,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance, verbose=args.verbose,
            n_trials=args.n_trials, n_jobs=args.n_jobs,
            reference=args.reference,
            random_state=random_state)

        kmeans_ds.attrs['input_file'] = args.input_file
        kmeans_ds.attrs['base_period_start_year'] = '{:d}'.format(clim_base_period[0])
        kmeans_ds.attrs['base_period_end_year'] = '{:d}'.format(clim_base_period[1])
        kmeans_ds.attrs['random_seed'] = '{:d}'.format(args.random_seed)

        kmeans_ds.to_netcdf(args.output_file)


if __name__ == '__main__':
    main()
