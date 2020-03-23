"""
Provides helper routines for k-means clustering.
"""

# License: MIT

from __future__ import absolute_import, division


import numpy as np

from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import check_random_state


def _calculate_uniform_reference_wk(X, n_clusters, n_init=10,
                                    n_jobs=None, random_state=None):
    rng = check_random_state(random_state)

    n_samples, n_features = X.shape

    feature_min = np.broadcast_to(np.min(X, axis=0), (n_samples, n_features))
    feature_max = np.broadcast_to(np.max(X, axis=0), (n_samples, n_features))

    random_data = ((feature_max - feature_min) * rng.uniform(
        size=(n_samples, n_features)) + feature_min)

    kmeans = KMeans(
        n_clusters=n_clusters, n_init=n_init,
        n_jobs=n_jobs, random_state=rng).fit(random_data)

    return kmeans.inertia_


def _calculate_pca_reference_wk(X, n_clusters, n_init=10,
                                n_components=100, n_iter=10, n_jobs=None,
                                random_state=None):
    rng = check_random_state(random_state)

    n_samples = X.shape[0]

    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter,
                       random_state=rng)
    svd.fit(X)

    Vh = svd.components_

    Xp = np.dot(X, np.transpose(Vh))

    feature_min = np.broadcast_to(np.min(Xp, axis=0), (n_samples, n_components))
    feature_max = np.broadcast_to(np.max(Xp, axis=0), (n_samples, n_components))

    random_data = ((feature_max - feature_min) * rng.uniform(
        size=(n_samples, n_components)) + feature_min)

    random_data = np.dot(random_data, Vh)

    kmeans = KMeans(
        n_clusters=n_clusters, n_init=n_init, n_jobs=n_jobs,
        random_state=rng).fit(random_data)

    return kmeans.inertia_


def _calculate_reference_wk(X, n_components, reference='uniform',
                            random_state=None):

    if reference == 'uniform':
        return _calculate_uniform_reference_wk(
            X, n_components, random_state=random_state)

    if reference == 'pca':
        return _calculate_pca_reference_wk(
            X, n_components, random_state=random_state)

    raise ValueError("unrecognized reference distribution '%s'" % reference)


def gap_statistic(X, Wk, n_components, n_trials=100,
                  reference='uniform', n_jobs=1, random_state=None):
    """Calculate gap statistic for k-means clustering."""

    rng = check_random_state(random_state)

    random_seeds = []
    for _ in range(n_trials):
        has_seed_already = True

        while has_seed_already:
            seed = rng.randint(np.iinfo(np.int32).max)
            if seed not in random_seeds:
                random_seeds.append(seed)
                has_seed_already = False

    result = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_reference_wk)(
            X, n_components, reference=reference, random_state=random_seeds[i])
        for i in range(n_trials))

    Wk_ref = np.array(result)
    lnWk_ref = np.log(Wk_ref)

    sk = np.std(lnWk_ref) * np.sqrt(1 + 1.0 / n_trials)
    gap = lnWk_ref.mean() - np.log(Wk)

    return gap, sk
