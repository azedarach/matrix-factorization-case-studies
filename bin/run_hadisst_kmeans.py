"""
Run k-means on HadISST SST anomalies.
"""

# License: MIT

from __future__ import absolute_import, division, print_function

import argparse
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import check_random_state
import xarray as xr

from convex_dim_red import gap_statistic

LAT_NAME = 'latitude'
LON_NAME = 'longitude'
TIME_NAME = 'time'
ANOMALY_NAME = 'sst_anom'
STD_ANOMALY_NAME = 'sst_std_anom'

# First and last years to retain for analysis
START_YEAR = 1870
END_YEAR = 2018

# Zonal extents of analysis region
MIN_LATITUDE = -45.5
MAX_LATITUDE = 45.5


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Run k-means on HadISST SST anomalies')

    parser.add_argument('input_file',
                        help='input file containing SST anomalies')
    parser.add_argument('output_file',
                        help='name of output file')
    parser.add_argument('--n-components', dest='n_components', type=int,
                        default=1, help='number of clusters')
    parser.add_argument('--init', dest='init', choices=['random', 'k-means++'],
                        default='k-means++', help='initialization method')
    parser.add_argument('--n-init', dest='n_init', type=int,
                        default=10, help='number of initializations')
    parser.add_argument('--lat-weights', dest='lat_weights',
                        choices=['none', 'cos', 'scos'], default='scos',
                        help='latitudinal weighting to apply')
    parser.add_argument('--tolerance', dest='tolerance', type=float,
                        default=1e-4, help='stopping tolerance')
    parser.add_argument('--max-iterations', dest='max_iterations', type=int,
                        default=10000, help='maximum number of iterations')
    parser.add_argument('--random-seed', dest='random_seed', type=int,
                        default=None, help='random seed')
    parser.add_argument('--cross-validate', dest='cross_validate',
                        action='store_true', help='use k-fold cross validation')
    parser.add_argument('--n-folds', dest='n_folds', type=int, default=10,
                        help='number of cross-validation folds')
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
    parser.add_argument('--standardized', dest='standardized',
                        action='store_true',
                        help='treat input data as standardized anomalies')

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


def get_latitude_weights(da, lat_weights='scos', lat_name=LAT_NAME):
    """Get latitude weights."""

    if lat_weights == 'cos':
        return np.cos(np.deg2rad(da[lat_name])).clip(0., 1.)

    if lat_weights == 'scos':
        return np.cos(np.deg2rad(da[lat_name])).clip(0., 1.) ** 0.5

    if lat_weights == 'none':
        return xr.ones_like(da[lat_name])

    raise ValueError("Invalid weights descriptor '%r'" % lat_weights)


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


def run_kmeans(da, n_components=2, lat_weights='none',
               init='k-means++', n_init=100,
               max_iterations=10000, tolerance=1e-4,
               verbose=False, random_state=None,
               cross_validate=False, n_folds=10,
               validation_frac=0.1, n_trials=100,
               reference='uniform', n_jobs=1,
               lat_name=LAT_NAME, sample_dim=TIME_NAME):
    """Run k-means on SST data."""

    rng = check_random_state(random_state)

    feature_dims = [d for d in da.dims if d != sample_dim]
    original_shape = [da.sizes[d] for d in da.dims if d != sample_dim]

    # Get requested latitude weights
    weights = get_latitude_weights(da, lat_weights=lat_weights,
                                   lat_name=lat_name)

    # Convert input data array to plain 2D array
    flat_data = weight_and_flatten_data(da, weights=weights, sample_dim=sample_dim)

    n_samples, n_features = flat_data.shape

    # Remove any features/columns with missing data
    missing_features = np.any(np.isnan(flat_data), axis=0)
    valid_data = flat_data[:, np.logical_not(missing_features)]

    # Split into training and validation sets
    n_training_samples = int(np.ceil((1 - validation_frac) * n_samples))
    n_validation_samples = n_samples - n_training_samples

    training_data = valid_data[:n_training_samples]
    validation_data = valid_data[n_training_samples:]
    training_samples = da[sample_dim].isel(
        {sample_dim: slice(0, n_training_samples)})

    if cross_validate:

        tscv = TimeSeriesSplit(n_splits=n_folds)

        training_gaps = []
        training_sks = []
        training_costs = []
        training_rmses = []
        test_costs = []
        test_rmses = []

        for train, test in tscv.split(training_data):

            kmeans_model, gap, sk = fit_kmeans_model(
                training_data[train], n_components=n_components,
                init=init, n_init=n_init, tolerance=tolerance,
                max_iterations=max_iterations, n_trials=n_trials,
                verbose=verbose, reference=reference, n_jobs=n_jobs,
                random_state=rng)

            reconstruction = np.empty_like(training_data[train])
            for i in range(n_components):
                mask = kmeans_model.labels_ == i
                reconstruction[mask] = kmeans_model.cluster_centers_[i]

            training_gaps.append(gap)
            training_sks.append(sk)
            training_costs.append(kmeans_model.inertia_)
            training_rmses.append(
                mean_squared_error(training_data[train], reconstruction,
                                   squared=False))

            test_labels = kmeans_model.predict(training_data[test])
            test_distances = kmeans_model.transform(training_data[test])
            test_cost = np.sum(np.min(test_distances ** 2), axis=1)

            reconstruction = np.empty_like(training_data[test])
            for i in range(n_components):
                mask = test_labels == i
                reconstruction[mask] = kmeans_model.cluster_centers_[i]

            test_costs.append(test_cost)
            test_rmses.append(
                mean_squared_error(training_data[test], reconstruction,
                                   squared=False))

        start_time = time.perf_counter()
        best_model, gap, sk = fit_kmeans_model(
            training_data, n_components=n_components,
            init=init, n_init=n_init, tolerance=tolerance,
            max_iterations=max_iterations, n_trials=n_trials,
            verbose=verbose, reference=reference, n_jobs=n_jobs,
            random_state=rng)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        kmeans_labels = np.zeros((n_training_samples, n_components))
        for i in range(n_components):
            mask = best_model.labels_ == i
            kmeans_labels[mask, i] = 1

        kmeans_labels_da = xr.DataArray(
            kmeans_labels,
            coords={sample_dim: training_samples,
                    'component': np.arange(n_components)},
            dims=[sample_dim, 'component'])

        valid_dictionary = best_model.cluster_centers_
        full_dictionary = np.full((n_components, n_features), np.NaN)
        full_dictionary[:, np.logical_not(missing_features), :] = valid_dictionary
        dictionary = np.reshape(
            full_dictionary, [n_components,] + original_shape)

        dictionary_coords = {d: da[d] for d in feature_dims}
        dictionary_coords['component'] = np.arange(n_components)

        dictionary_dims = ['component'] + feature_dims

        kmeans_dictionary_da = xr.DataArray(
            dictionary, coords=dictionary_coords, dims=dictionary_dims)

        validation_labels = best_model.predict(validation_data)
        validation_distances = best_model.transform(validation_data)
        validation_cost = np.sum(np.min(validation_distances ** 2, axis=1))

        reconstruction = np.empty_like(validation_data)
        for i in range(n_components):
            mask = validation_labels == i
            reconstruction[mask] = best_model.cluster_centers_[i]

        validation_rmse = mean_squared_error(
            validation_data, reconstruction, squared=False)

        data_vars = {'weights': kmeans_labels_da,
                     'dictionary': kmeans_dictionary_da}

        kmeans_ds = xr.Dataset(data_vars)

        kmeans_ds.attrs['training_set_cost'] = '{:16.8e}'.format(np.mean(test_costs))
        kmeans_ds.attrs['training_set_cost_std'] = '{:16.8e}'.format(np.std(test_costs))
        kmeans_ds.attrs['training_set_rmse'] = '{:16.8e}'.format(np.mean(test_rmses))
        kmeans_ds.attrs['training_set_rmse_std'] = '{:16.8e}'.format(np.std(test_rmses))
        kmeans_ds.attrs['training_set_gap'] = '{:16.8e}'.format(np.mean(training_gaps))
        kmeans_ds.attrs['training_set_gap_std'] = '{:16.8e}'.format(np.std(training_gaps))
        kmeans_ds.attrs['training_set_sk'] = '{:16.8e}'.format(np.mean(training_sks))
        kmeans_ds.attrs['training_set_sk_std'] = '{:16.8e}'.format(np.std(training_sks))
        kmeans_ds.attrs['training_set_size'] = '{:d}'.format(n_training_samples)

        kmeans_ds.attrs['test_set_cost'] = '{:16.8e}'.format(validation_cost)
        kmeans_ds.attrs['test_set_size'] = '{:d}'.format(n_validation_samples)
        kmeans_ds.attrs['test_set_rmse'] = '{:16.8e}'.format(validation_rmse)

        kmeans_ds.attrs['gap_statistic'] = '{:16.8e}'.format(gap)
        kmeans_ds.attrs['gap_sk'] = '{:16.8e}'.format(sk)
        kmeans_ds.attrs['n_folds'] = '{:d}'.format(n_folds)
        kmeans_ds.attrs['n_iter'] = '{:d}'.format(best_model.n_iter_)

    else:

        start_time = time.perf_counter()
        best_model, gap, sk = fit_kmeans_model(
            training_data, n_components=n_components,
            init=init, n_init=n_init, tolerance=tolerance,
            max_iterations=max_iterations, n_trials=n_trials,
            verbose=verbose, reference=reference, n_jobs=n_jobs,
            random_state=rng)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        reconstruction = np.empty_like(training_data)
        for i in range(n_components):
            mask = best_model.labels_ == i
            reconstruction[mask] = best_model.cluster_centers_[i]

        training_cost = best_model.inertia_
        training_rmse = mean_squared_error(training_data, reconstruction, squared=False)

        kmeans_labels = np.zeros((n_training_samples, n_components))
        for i in range(n_components):
            mask = best_model.labels_ == i
            kmeans_labels[mask, i] = 1

        kmeans_labels_da = xr.DataArray(
            kmeans_labels,
            coords={sample_dim: training_samples,
                    'component': np.arange(n_components)},
            dims=[sample_dim, 'component'])

        valid_dictionary = best_model.cluster_centers_
        full_dictionary = np.full((n_components, n_features), np.NaN)
        full_dictionary[:, np.logical_not(missing_features)] = valid_dictionary
        dictionary = np.reshape(
            full_dictionary, [n_components,] + original_shape)

        dictionary_coords = {d: da[d] for d in feature_dims}
        dictionary_coords['component'] = np.arange(n_components)

        dictionary_dims = ['component'] + feature_dims

        kmeans_dictionary_da = xr.DataArray(
            dictionary, coords=dictionary_coords, dims=dictionary_dims)

        validation_labels = best_model.predict(validation_data)
        validation_distances = best_model.transform(validation_data)
        validation_cost = np.sum(np.min(validation_distances ** 2, axis=1))

        reconstruction = np.empty_like(validation_data)
        for i in range(n_components):
            mask = validation_labels == i
            reconstruction[mask] = best_model.cluster_centers_[i]

        validation_rmse = mean_squared_error(
            validation_data, reconstruction, squared=False)

        data_vars = {'weights': kmeans_labels_da,
                     'dictionary': kmeans_dictionary_da}

        kmeans_ds = xr.Dataset(data_vars)

        kmeans_ds.attrs['training_set_cost'] = '{:16.8e}'.format(training_cost)
        kmeans_ds.attrs['training_set_rmse'] = '{:16.8e}'.format(training_rmse)
        kmeans_ds.attrs['training_set_size'] = '{:d}'.format(n_training_samples)

        kmeans_ds.attrs['test_set_cost'] = '{:16.8e}'.format(validation_cost)
        kmeans_ds.attrs['test_set_rmse'] = '{:16.8e}'.format(validation_rmse)
        kmeans_ds.attrs['test_set_size'] = '{:d}'.format(n_validation_samples)

        kmeans_ds.attrs['gap_statistic'] = '{:16.8e}'.format(gap)
        kmeans_ds.attrs['gap_sk'] = '{:16.8e}'.format(sk)
        kmeans_ds.attrs['n_iter'] = '{:d}'.format(best_model.n_iter_)

    kmeans_ds.attrs['lat_weights'] = lat_weights
    kmeans_ds.attrs['init'] = init
    kmeans_ds.attrs['n_init'] = '{:d}'.format(n_init)
    kmeans_ds.attrs['max_iterations'] = '{:d}'.format(max_iterations)
    kmeans_ds.attrs['tolerance'] = '{:16.8e}'.format(tolerance)
    kmeans_ds.attrs['reference'] = '{}'.format(reference)
    kmeans_ds.attrs['n_trials'] = '{:d}'.format(n_trials)
    kmeans_ds.attrs['validation_frac'] = '{:16.8e}'.format(validation_frac)
    kmeans_ds.attrs['elapsed_time'] = '{:16.8e}'.format(elapsed_time)

    return kmeans_ds


def main():
    """Run k-means on HadISST SST anomalies."""

    args = parse_cmd_line_args()

    random_state = check_random_state(args.random_seed)

    if args.standardized:
        var_name = STD_ANOMALY_NAME
    else:
        var_name = ANOMALY_NAME

    with xr.open_dataset(args.input_file) as sst_anom_ds:

        sst_anom_ds = sst_anom_ds.where(
            (sst_anom_ds[TIME_NAME].dt.year >= START_YEAR) &
            (sst_anom_ds[TIME_NAME].dt.year <= END_YEAR), drop=True)

        sst_anom_ds = sst_anom_ds.where(
            (sst_anom_ds[LAT_NAME] >= MIN_LATITUDE) &
            (sst_anom_ds[LAT_NAME] <= MAX_LATITUDE), drop=True)

        clim_base_period = [int(sst_anom_ds.attrs['base_period_start_year']),
                            int(sst_anom_ds.attrs['base_period_end_year'])]

        sst_anom_da = sst_anom_ds[var_name]

        if args.restrict_to_base_period:
            sst_anom_da = sst_anom_da.where(
                (sst_anom_da[TIME_NAME].dt.year >= clim_base_period[0]) &
                (sst_anom_da[TIME_NAME].dt.year <= clim_base_period[1]), drop=True)

        kmeans_ds = run_kmeans(
            sst_anom_da, n_components=args.n_components,
            lat_weights=args.lat_weights, init=args.init, n_init=args.n_init,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance, cross_validate=args.cross_validate,
            n_folds=args.n_folds, verbose=args.verbose,
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
