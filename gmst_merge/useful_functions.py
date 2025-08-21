#  Global Temperature Merge - a package for merging global temperature datasets.
#  Copyright \(c\) 2025 John Kennedy and Bruce Calvert
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Code written by Bruce T. T. Calvert - August 2025

import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


def balanced_kmeans(
        ensemble,
        number_of_clusters,
        rng=np.random.default_rng(seed=0),
        weights=["null"]
):

    ensemble = np.array(ensemble)
    if round(number_of_clusters, 0) != number_of_clusters or number_of_clusters < 1:
        raise TypeError("number of clusters must be a positive integer")
    if ensemble.ndim == 1:
        ensemble = ensemble.reshape((-1, 1))
    if ensemble.ndim != 2:
        raise TypeError("ensemble must be a 2 dimensional array")
    if ensemble.shape[1] % number_of_clusters != 0:
        raise TypeError("number of ensemble members must be divisible by number of clusters")
    if weights[0] != "null":
        weights = np.array(weights)
        weights = weights.reshape(weights.size)
    if weights[0] == "null":
        weights = np.ones(ensemble.shape[0])
    if weights.shape[0] != ensemble.shape[0]:
        raise TypeError("weights dimensions must agree with number of time periods of hierarchy")
    if weights.shape[0] != weights.size:
        raise TypeError("weights must be a 1 dimensional vector")

    cluster_size = int(ensemble.shape[1] / number_of_clusters)
    ensemble = np.transpose(ensemble)

    # Assign weights
    ensemble_mean = np.mean(ensemble, axis=0)
    ensemble = np.multiply(ensemble - ensemble_mean, np.sqrt(weights)) + ensemble_mean

    # Initialize clusters using K means algorithm
    kmeans = KMeans(
        n_clusters=min(number_of_clusters, np.unique(ensemble, axis=0).shape[0]),
        random_state=rng.integers((1 << 31) - 1)
    ).fit(ensemble)

    cluster_centers = kmeans.cluster_centers_

    # Adjustment for case where there were insufficient unique ensemble members
    cluster_centers = np.append(
        np.kron(np.ones(((int(number_of_clusters / cluster_centers.shape[0]), 1))), cluster_centers),
        cluster_centers[:np.mod(number_of_clusters, cluster_centers.shape[0]), :],
        axis=0
    )

    # Iteratively assign equal numbers of ensemble members to each cluster using Hungarian-like
    # algorithm and recalculate cluster centers
    distance_matrix = np.zeros((ensemble.shape[0], number_of_clusters))
    old_assigned_cluster = np.zeros((ensemble.shape[0]))
    old_square_distance = np.zeros((1))
    for iterations in range(1, 100):
        for i in range(ensemble.shape[0]):
            for j in range(number_of_clusters):
                distance_matrix[i, j] = np.linalg.norm(ensemble[i] - cluster_centers[j])

        row_index, column_index = linear_sum_assignment(np.kron(np.ones((1, int(cluster_size))), distance_matrix))
        assigned_cluster = column_index % number_of_clusters
        square_distance = np.zeros((1))

        for i in range(number_of_clusters):
            cluster_centers[i] = ensemble[assigned_cluster == i].mean(axis=0)

        for i in range(ensemble.shape[0]):
            square_distance += distance_matrix[i, assigned_cluster[i]]

        # Have we converged yet?
        if (assigned_cluster == old_assigned_cluster).all() or square_distance == old_square_distance:
            print(iterations, np.mean((assigned_cluster - old_assigned_cluster) ** 2))
            break
        else:
            print(iterations, np.mean((assigned_cluster - old_assigned_cluster) ** 2))
            old_assigned_cluster = assigned_cluster
            old_square_distance = square_distance

    return assigned_cluster, cluster_centers


def gridded_to_timeseries(array):
    array = np.array(array)
    if array.ndim == 4:
        if array.shape[3] == 1:
            array = array.reshape((array.shape[0], array.shape[1], array.shape[2]))

    if array.ndim < 3:
        raise TypeError("array must be a 3 dimensional array")

    # assumes an equirectangular grid where 1st dimension is longitude and 2nd dimension is latitude
    weight = (
                np.cos(np.arange(array.shape[1]) / array.shape[1] * np.pi) -
                np.cos((np.arange(array.shape[1]) + 1) / array.shape[1] * np.pi)
    ) / 2

    weight = np.transpose(
        np.tile(weight, (array.shape[2], array.shape[0], 1)), [1, 2, 0]
    ) / array.shape[0]

    timeseries = np.divide(
        np.nansum(np.nansum(np.multiply(array, weight), axis=0), axis=0),
              np.nansum(np.nansum(np.multiply(1 - np.isnan(array).astype(int), weight), axis=0), axis=0)
    )

    return timeseries


def days_in_month(years):
    """Get number of days in the months of a specified year"""
    years = np.array(years)
    years = np.round(np.reshape(years.copy(), (1, years.size), order='F'))

    month_lengths = np.transpose([[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]])
    leap_offsets = np.transpose([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    is_leap_year = (
            (years % 4 == 0).astype(int) - (years % 100 == 0).astype(int) + (years % 400 == 0).astype(int)
    )

    output = month_lengths + leap_offsets * is_leap_year

    return output


def monthly_to_annual_timeseries(matrix, first_year):

    if round(first_year, 0) != first_year:
        raise TypeError("first year must be an integer")

    matrix = np.array(matrix)

    if matrix.ndim == 1:
        matrix = matrix.reshape((-1, 1))

    if matrix.ndim != 2:
        raise TypeError("first input must be a 2 dimensional array")

    matrix_reshaped = matrix[:(matrix.shape[0] // 12) * 12].reshape(12, matrix.shape[0] // 12, matrix.shape[1], order='F')
    years = np.arange(first_year, first_year + matrix.shape[0] // 12)
    month_weights = days_in_month(years).astype(float)
    month_weights = np.divide(month_weights, np.sum(month_weights, axis=0)).reshape(12, matrix.shape[0] // 12, 1,
                                                                                    order='F')
    annualized_matrix = np.sum(np.multiply(matrix_reshaped, month_weights), axis=0).reshape(-1, matrix.shape[1])

    return annualized_matrix
