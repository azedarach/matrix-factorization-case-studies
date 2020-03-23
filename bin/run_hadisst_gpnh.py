"""
Run GPNH-regularized convex coding on HadISST SST anomalies.
"""

# License: MIT

from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
import time

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import check_random_state
import xarray as xr

from convex_dim_red import GPNHConvexCoding


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

LAT_WEIGHTS = 'scos'

# Fraction of data used for assessing out of sample reconstruction error
VALIDATION_FRAC = 0.1

# Number of random restarts to use
INIT = 'random'
N_INIT = 100

# Stopping criteria
MAX_ITERATIONS = 10000
TOLERANCE = 1e-6


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Run GPNH-regularized convex coding on HadISST SST anomalies'
    )

    parser.add_argument('input_file',
                        help='input file containing SST anomalies')
    parser.add_argument('output_file',
                        help='name of output file')
    parser.add_argument('--n-components', dest='n_components', type=int,
                        default=1, help='number of clusters')
    parser.add_argument('--lambda-W', dest='lambda_W', type=float,
                        default=0.0, help='dictionary regularization')
    parser.add_argument('--init', dest='init', choices=['random', 'furthest_sum'],
                        default=INIT, help='initialization method')
    parser.add_argument('--n-init', dest='n_init', type=int,
                        default=N_INIT, help='number of initializations')
    parser.add_argument('--lat-weights', dest='lat_weights',
                        choices=['none', 'cos', 'scos'], default=LAT_WEIGHTS,
                        help='latitudinal weighting to apply')
    parser.add_argument('--tolerance', dest='tolerance', type=float,
                        default=TOLERANCE, help='stopping tolerance')
    parser.add_argument('--max-iterations', dest='max_iterations', type=int,
                        default=MAX_ITERATIONS, help='maximum number of iterations')
    parser.add_argument('--random-seed', dest='random_seed', type=int,
                        default=None, help='random seed')
    parser.add_argument('--cross-validate', dest='cross_validate',
                        action='store_true', help='use k-fold cross validation')
    parser.add_argument('--n-folds', dest='n_folds', type=int, default=10,
                        help='number of cross-validation folds')
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

    if args.lambda_W < 0:
        raise ValueError('Dictionary regularization must be non-negative')

    if args.tolerance < 0:
        raise ValueError('Stopping tolerance must be positive')

    if args.n_init < 1:
        raise ValueError('Number of initializations must be at least 1')

    if args.max_iterations < 1:
        raise ValueError('Maximum number of iterations must be at least 1')

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


def fit_gpnh_model(X, n_components=2, lambda_W=0, init=INIT, n_init=N_INIT,
                   tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS, verbose=False,
                   random_state=None, **kwargs):
    """Run GPNH convex coding on given data."""

    rng = check_random_state(random_state)

    min_cost = None
    best_model = None
    for _ in range(n_init):

        model = GPNHConvexCoding(n_components=n_components, lambda_W=lambda_W,
                                 init=init, tolerance=tolerance,
                                 max_iterations=max_iterations, verbose=verbose,
                                 random_state=rng, **kwargs)

        model.fit_transform(X)

        if min_cost is None or model.cost < min_cost:
            best_model = deepcopy(model)
            min_cost = model.cost

    return best_model


def run_gpnh(da, n_components=2, lambda_W=0, lat_weights=LAT_WEIGHTS,
             init=INIT, n_init=N_INIT,
             max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE,
             verbose=False, random_state=None,
             cross_validate=False, n_folds=10,
             validation_frac=VALIDATION_FRAC,
             lat_name=LAT_NAME, sample_dim=TIME_NAME):
    """Run GPNH convex coding on SST data."""

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

    # Divide data into training and validation sets
    n_training_samples = int(np.ceil((1 - validation_frac) * n_samples))
    n_validation_samples = n_samples - n_training_samples

    training_data = valid_data[:n_training_samples]
    validation_data = valid_data[n_training_samples:]

    training_samples = da[sample_dim].isel({sample_dim: slice(0, n_training_samples)})

    if cross_validate:

        tscv = TimeSeriesSplit(n_splits=n_folds)

        training_costs = []
        training_rmses = []
        test_costs = []
        test_rmses = []

        for train, test in tscv.split(training_data):

            gpnh_model = fit_gpnh_model(
                training_data[train], n_components=n_components, lambda_W=lambda_W,
                init=init, n_init=n_init, tolerance=tolerance,
                max_iterations=max_iterations,
                verbose=verbose, random_state=rng)

            reconstruction = gpnh_model.inverse_transform(gpnh_model.weights)

            training_costs.append(gpnh_model.cost)
            training_rmses.append(
                mean_squared_error(training_data[train], reconstruction,
                                   squared=False))

            test_weights, test_cost = gpnh_model.transform(training_data[test])

            reconstruction = gpnh_model.inverse_transform(test_weights)

            test_costs.append(test_cost)
            test_rmses.append(
                mean_squared_error(training_data[test], reconstruction,
                                   squared=False))

        start_time = time.perf_counter()
        best_model = fit_gpnh_model(
            training_data, n_components=n_components, lambda_W=lambda_W,
            init=init, n_init=n_init,
            tolerance=tolerance, max_iterations=max_iterations, verbose=verbose,
            random_state=rng)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        gpnh_weights_da = xr.DataArray(
            best_model.weights, coords={sample_dim: training_samples,
                                        'component': np.arange(n_components)},
            dims=[sample_dim, 'component'])

        valid_dictionary = best_model.dictionary.T
        full_dictionary = np.full((n_components, n_features), np.NaN)
        full_dictionary[:, np.logical_not(missing_features), :] = valid_dictionary
        dictionary = np.reshape(
            full_dictionary, [n_components,] + original_shape)

        dictionary_coords = {d: da[d] for d in feature_dims}
        dictionary_coords['component'] = np.arange(n_components)

        dictionary_dims = ['component'] + feature_dims

        gpnh_dictionary_da = xr.DataArray(
            dictionary, coords=dictionary_coords, dims=dictionary_dims)

        cost_deltas_da = xr.DataArray(
            best_model.cost_deltas,
            coords={'iteration': np.arange(best_model.n_iter)},
            dims=['iteration'])

        validation_weights, validation_cost = best_model.transform(validation_data)

        reconstruction = best_model.inverse_transform(validation_weights)
        validation_rmse = mean_squared_error(validation_data, reconstruction, squared=False)

        data_vars = {'weights': gpnh_weights_da,
                     'dictionary': gpnh_dictionary_da,
                     'cost_deltas': cost_deltas_da}

        gpnh_ds = xr.Dataset(data_vars)

        gpnh_ds.attrs['training_set_cost'] = '{:16.8e}'.format(np.mean(test_costs))
        gpnh_ds.attrs['training_set_cost_std'] = '{:16.8e}'.format(np.std(test_costs))
        gpnh_ds.attrs['training_set_rmse'] = '{:16.8e}'.format(np.mean(test_rmses))
        gpnh_ds.attrs['training_set_rmse_std'] = '{:16.8e}'.format(np.std(test_rmses))
        gpnh_ds.attrs['training_set_size'] = '{:d}'.format(n_training_samples)

        gpnh_ds.attrs['test_set_cost'] = '{:16.8e}'.format(validation_cost)
        gpnh_ds.attrs['test_set_size'] = '{:d}'.format(n_validation_samples)
        gpnh_ds.attrs['test_set_rmse'] = '{:16.8e}'.format(validation_rmse)

        gpnh_ds.attrs['n_folds'] = '{:d}'.format(n_folds)
        gpnh_ds.attrs['n_iter'] = '{:d}'.format(best_model.n_iter)
        gpnh_ds.attrs['avg_time_per_iter'] = '{:16.8e}'.format(
            best_model.avg_time_per_iter)

    else:

        start_time = time.perf_counter()
        best_model = fit_gpnh_model(
            training_data, n_components=n_components, lambda_W=lambda_W,
            init=init, n_init=n_init,
            tolerance=tolerance, max_iterations=max_iterations, verbose=verbose,
            random_state=rng)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        reconstruction = best_model.inverse_transform(best_model.weights)

        training_cost = best_model.cost
        training_rmse = mean_squared_error(training_data, reconstruction, squared=False)

        gpnh_weights_da = xr.DataArray(
            best_model.weights, coords={sample_dim: training_samples,
                                        'component': np.arange(n_components)},
            dims=[sample_dim, 'component'])

        valid_dictionary = best_model.dictionary.T
        full_dictionary = np.full((n_components, n_features), np.NaN)
        full_dictionary[:, np.logical_not(missing_features)] = valid_dictionary
        dictionary = np.reshape(
            full_dictionary, [n_components,] + original_shape)

        dictionary_coords = {d: da[d] for d in feature_dims}
        dictionary_coords['component'] = np.arange(n_components)

        dictionary_dims = ['component'] + feature_dims

        gpnh_dictionary_da = xr.DataArray(
            dictionary, coords=dictionary_coords, dims=dictionary_dims)

        cost_deltas_da = xr.DataArray(
            best_model.cost_deltas,
            coords={'iteration': np.arange(best_model.n_iter + 1)},
            dims=['iteration'])

        validation_weights, validation_cost = best_model.transform(validation_data)
        reconstruction = best_model.inverse_transform(validation_weights)

        validation_rmse = mean_squared_error(validation_data, reconstruction, squared=False)

        data_vars = {'weights': gpnh_weights_da,
                     'dictionary': gpnh_dictionary_da,
                     'cost_deltas': cost_deltas_da}

        gpnh_ds = xr.Dataset(data_vars)

        gpnh_ds.attrs['training_set_cost'] = '{:16.8e}'.format(training_cost)
        gpnh_ds.attrs['training_set_rmse'] = '{:16.8e}'.format(training_rmse)
        gpnh_ds.attrs['training_set_size'] = '{:d}'.format(n_training_samples)

        gpnh_ds.attrs['test_set_cost'] = '{:16.8e}'.format(validation_cost)
        gpnh_ds.attrs['test_set_rmse'] = '{:16.8e}'.format(validation_rmse)
        gpnh_ds.attrs['test_set_size'] = '{:d}'.format(n_validation_samples)

        gpnh_ds.attrs['n_iter'] = '{:d}'.format(best_model.n_iter)
        gpnh_ds.attrs['avg_time_per_iter'] = '{:16.8e}'.format(
            best_model.avg_time_per_iter)

    gpnh_ds.attrs['lat_weights'] = lat_weights
    gpnh_ds.attrs['init'] = init
    gpnh_ds.attrs['n_init'] = '{:d}'.format(n_init)
    gpnh_ds.attrs['lambda_W'] = '{:16.8e}'.format(lambda_W)
    gpnh_ds.attrs['max_iterations'] = '{:d}'.format(max_iterations)
    gpnh_ds.attrs['tolerance'] = '{:16.8e}'.format(tolerance)
    gpnh_ds.attrs['elapsed_time'] = '{:16.8e}'.format(elapsed_time)

    return gpnh_ds


def main():
    """Run GPNH-regularized convex coding on HadISST SST anomalies."""

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

        gpnh_ds = run_gpnh(
            sst_anom_da, n_components=args.n_components, lambda_W=args.lambda_W,
            lat_weights=args.lat_weights, init=args.init, n_init=args.n_init,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance, cross_validate=args.cross_validate,
            n_folds=args.n_folds, verbose=args.verbose,
            random_state=random_state)

        gpnh_ds.attrs['input_file'] = args.input_file
        gpnh_ds.attrs['base_period_start_year'] = '{:d}'.format(clim_base_period[0])
        gpnh_ds.attrs['base_period_end_year'] = '{:d}'.format(clim_base_period[1])
        gpnh_ds.attrs['random_seed'] = '{:d}'.format(args.random_seed)

        gpnh_ds.to_netcdf(args.output_file)


if __name__ == '__main__':
    main()
