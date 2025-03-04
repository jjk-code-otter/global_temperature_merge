#  Global Temperature Merge - a package for merging global temperature datasets.
#  Copyright \(c\) 2025 John Kennedy
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

import copy
import numpy as np
import pandas as pd
from random import randrange, choice
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


class Dataset:
    """
    Simple data class using numpy arrays for the time axis and a data ensemble.
    """

    def __init__(self, inarr, name='none'):
        self.data = inarr[:, 1:]
        self.time = inarr[:, 0]
        self.n_time = len(self.time)
        self.n_ensemble = self.data.shape[1]
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @staticmethod
    def read_csv(name, data_dir, header=None):
        """
        Read a file from the data directory

        :param name: str
            Name of the dataset
        :param data_dir: str
            Path of the data directory
        :return: Dataset
        """
        filestub = 'ensemble_time_series.csv'
        df = pd.read_csv(data_dir / name / filestub, header=header)
        df = df.to_numpy()
        return Dataset(df, name=name)

    def read_csv_from_file(filename, name, header=None):
        """
        Read a file from the data directory

        :param name: str
            Name of the dataset
        :param data_dir: str
            Path of the data directory
        :return: Dataset
        """
        df = pd.read_csv(filename, header=header)
        df = df.to_numpy()
        return Dataset(df, name=name)

    @staticmethod
    def join(tail, head, join_start_year, join_end_year):
        """
        Join two Datasets together. The Dataset is made by joining a tail Dataset to a head Dataset. This is
        done by anomalizing each Dataset to a common period specified by join_start_year and join_end_year and then
        cutting each Dataset at the midpoint of the common period and splicing them together.

        :param tail: Dataset
            Tail dataset
        :param head: Dataset
            Head dataset
        :param join_start_year: int
            First year for the common period used for splicing
        :param join_end_year: int
            Last year for the common period used for splicing
        :return: Dataset
            The spliced dataset.
        """

        mid_point = int((join_end_year + join_start_year) / 2)

        tail.anomalize(join_start_year, join_end_year)
        head.anomalize(join_start_year, join_end_year)

        first_year = tail.get_start_year()
        last_year = head.get_end_year()
        head_first_year = head.get_start_year()

        out_n_time = last_year - first_year + 1

        out_arr = np.zeros((out_n_time, 2))

        out_arr[0: mid_point - first_year + 1, 0] = tail.time[0: mid_point - first_year + 1]
        out_arr[0: mid_point - first_year + 1, 1] = tail.data[0: mid_point - first_year + 1, 0]

        out_arr[mid_point - first_year + 1:, 0] = head.time[mid_point - head_first_year + 1:]
        out_arr[mid_point - first_year + 1:, 1] = head.data[mid_point - head_first_year + 1:, 0]

        return Dataset(out_arr, 'meta_ensemble')

    def sample_from_ensemble(self, rng):
        """
        Pick a single ensemble member from the Dataset and return it as a Dataset

        :return: Dataset
            A single ensemble member
        """
        if self.n_ensemble == 1:
            return copy.deepcopy(self)
        else:
            selected_member = rng.integers(self.n_ensemble)
            out_arr = np.zeros((self.n_time, 2))
            out_arr[:, 0] = self.time[:]
            out_arr[:, 1] = self.data[:, selected_member]
            return Dataset(out_arr, name=f'{self.name} {selected_member:04d}')

    def get_start_year(self):
        """
        Get the earliest year in the Dataset
        :return:
        """
        return int(np.min(self.time))

    def get_end_year(self):
        """
            Get the latest year in the Dataset
            :return:
        """
        return int(np.max(self.time))

    def get_quantile_range(self, percent_range):
        """
        Get the quantiles at each time point that correspond to a particular percent range. e.g. the 95% range is
        between the 2.5 perctile and the 97.5 percentile.

        :param percent_range: float
            The percentage of the distribution for which the quantiles should be calculated.
        :return: np.ndarray, np.ndarray
            The lower and upper quantile bounds
        """
        fraction = percent_range / 100.
        remainder = (1 - fraction) / 2
        return np.quantile(self.data, remainder, axis=1), np.quantile(self.data, 1 - remainder, axis=1)

    def get_ensemble_mean(self):
        """
        Calculate the mean of all the ensemble members.

        :return: np.ndarray
        """
        return np.mean(self.data, axis=1)

    def get_ensemble_std(self):
        """
        Calculate the mean of all the ensemble members.

        :return: np.ndarray
        """
        return np.std(self.data, axis=1)

    def anomalize(self, in_start_year, in_end_year):
        """
        Calculate anomalies from the input array using the period from in_start_year to in_end_year to calculate the
        climatology.

        :param in_arr: np.ndarray
            Array containing the ensemble
        :param in_start_year: int
            First year of climatology period
        :param in_end_year: int
            Last year of climatology period
        :return: np.ndarray
        """
        # Calculate anomalies
        mask = (self.time >= in_start_year) & (self.time <= in_end_year)
        self.data = self.data - np.mean(self.data[mask, :], axis=0)

    def convert_to_perturbations(self):
        ensemble_mean = self.get_ensemble_mean()
        self.data = self.data - np.reshape(ensemble_mean, (self.n_time, 1))

    def convert_to_standardised_perturbations(self):
        ensemble_mean = self.get_ensemble_mean()
        ensemble_std = self.get_ensemble_std()
        self.data = self.data - np.reshape(ensemble_mean, (self.n_time, 1))
        self.data = self.data / np.reshape(ensemble_std, (self.n_time, 1))

    def make_perturbed_dataset(self, perturbation, scaling=None):
        if self.n_ensemble != 1:
            raise RuntimeError('Trying to add perturbations to an ensemble dataset')
        y1 = self.get_start_year()
        y2 = self.get_end_year()
        py1 = perturbation.get_start_year()
        py2 = perturbation.get_end_year()
        if py1 > y1 or py2 < y2:
            raise RuntimeError('Perturbations do not cover whole dataset period')

        out_ds = copy.deepcopy(self)

        if scaling is None:
            out_ds.data = np.reshape(out_ds.data, (out_ds.n_time, 1)) + perturbation.data[y1 - py1:y2 - py1 + 1, :]
        else:
            n = perturbation.n_ensemble
            expanded_scaling = np.tile(scaling[:, 1:], (1, n))
            out_ds.data = np.reshape(out_ds.data, (out_ds.n_time, 1)) + expanded_scaling * perturbation.data[
                                                                                           y1 - py1:y2 - py1 + 1, :]

        out_ds.n_ensemble = perturbation.n_ensemble

        out_ds.name = f'{self.name}_{perturbation.name}'

        return out_ds

    def lowess_smooth(self):
        """
        Apply a lowess smoother to all ensemble members

        :return: Dataset
            Dataset containing smoothed ensemble members
        """
        lowess = sm.nonparametric.lowess
        smoothed_ensemble = copy.deepcopy(self)
        for i in range(self.n_ensemble):
            z = lowess(self.data[:, i], self.time, frac=1. / 5., return_sorted=False)
            smoothed_ensemble.data[:, i] = z[:]

        return smoothed_ensemble

    def thin_ensemble(self, n_thinned):

        # calculate long term change
        early_average = np.mean(self.data[0:51, :], axis=0)
        late_average = np.mean(self.data[-10:, :], axis=0)

        long_term_change = late_average - early_average
        order = np.argsort(long_term_change)

        out_data = np.zeros((self.n_time, n_thinned + 1))

        n_per_bin = self.n_ensemble / n_thinned

        out_data[:, 0] = self.time[:]
        for i in range(n_thinned):
            index = int((i * n_per_bin) + (n_per_bin / 2))
            out_data[:, i + 1] = self.data[:, order[index]]

        return Dataset(out_data, name=self.name)

    def cluster_ensemble(self, cluster_size, rng):

        ensemble = self.data

        if round(cluster_size, 0) != cluster_size or cluster_size < 1:
            raise TypeError("cluster size must be a positive integer")
        if ensemble.ndim == 1:
            ensemble = ensemble.reshape((-1, 1))
        if ensemble.ndim != 2:
            raise TypeError("ensemble must be a 2 dimensional array")
        if ensemble.shape[1] % cluster_size != 0:
            raise TypeError("number of ensemble members must be divisible by cluster size")

        number_of_clusters = int(ensemble.shape[1] / cluster_size)
        ensemble = np.transpose(ensemble)

        # Initialize clusters using K means algorithm
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=np.random.randint((1 << 31) - 1)).fit(ensemble)
        cluster_centers = kmeans.cluster_centers_

        # Iteratively assign equal numbers of ensemble members to each cluster using Hungarian-like
        # algorithm and recalculate cluster centers
        for iterations in range(1, 100):
            distance_matrix = np.zeros((ensemble.shape[0], number_of_clusters))
            for i in range(ensemble.shape[0]):
                for j in range(number_of_clusters):
                    distance_matrix[i, j] = np.linalg.norm(ensemble[i] - cluster_centers[j])
            row_index, column_index = linear_sum_assignment(np.kron(np.ones((1, int(cluster_size))), distance_matrix))
            assigned_cluster = column_index % number_of_clusters
            for i in range(number_of_clusters):
                cluster_centers[i] = ensemble[assigned_cluster == i].mean(axis=0)

        out_ensemble = copy.deepcopy(self)
        out_ensemble.data = np.transpose(cluster_centers)
        out_ensemble.n_time = len(out_ensemble.time)
        out_ensemble.n_ensemble = out_ensemble.data.shape[1]

        return out_ensemble

    def to_csv(self, filename, header=False) -> None:
        """
        Write dataset to csv file

        :param filename: str
            File path
        :return: None
        """
        df_full_ensemble = pd.DataFrame(data=self.data.astype(np.float16), index=self.time.astype(int))
        df_full_ensemble.to_csv(filename, sep=',', encoding='utf-8', header=header)

    def summary_to_csv(self, filename) -> None:
        """
        Write the summary statistics to csv file

        :param filename: str
            File path
        :return: None
        """
        mn = self.get_ensemble_mean().astype(np.float16)
        sd = self.get_ensemble_std().astype(np.float16)
        tm = self.time.astype(np.int)
        df = pd.DataFrame({'time': tm, 'mean': mn, 'std': sd})
        df.to_csv(filename, sep=',', encoding='utf-8')

    def plot_heat_map(self, filename, normalize=True) -> None:
        """
        Plot a heatmap showing the ensemble density as a function of anomaly (x-axis) and time (y-axis)

        :param filename: str
            Path of filename to write the plot to.
        :param normalize: bool
            If set to True each year is normalized separately. If set to False, normalization is done globally
        :return: None
        """

        y1 = self.get_start_year()
        y2 = self.get_end_year()

        bins = np.arange(-0.5, 1.77, 0.01)
        hmap = np.zeros((y2 - y1 + 1, len(bins) - 1))

        for y in range(y1, y2 + 1):
            n_plot_years = y2 - y1 + 1
            bins = np.arange(-0.5, 1.77, 0.01)
            h, b = np.histogram(self.data[y - y1, :], bins=bins)
            b = (b[0:-1] + b[1:]) / 2.
            h_prime = h / self.n_ensemble
            if normalize and np.max(h_prime) > 0:
                hmap[y - y1, :] = h_prime / np.max(h_prime)
            else:
                hmap[y - y1, :] = h_prime

        max_value = np.max(hmap)
        # hmap[hmap==0] = np.nan
        plt.figure(figsize=[16, 16])
        times = self.time
        # cmaps = ['terrain','gist_earth','cubehelix','turbo', 'flag', 'prism', 'twilight_shifted','copper']
        # chosen = choice(cmaps)
        # print(chosen)
        plt.pcolormesh(b, times, hmap, cmap='pink', vmin=0, vmax=max_value, shading='nearest')
        plt.gca().tick_params(axis='both', which='major', labelsize=20)
        plt.gca().invert_yaxis()
        plt.savefig(filename, dpi=300)
        plt.close()

    def get_exceedance_year(self, threshold):
        """
        For each ensemble member, find the first year at or above the specified threshold

        :param threshold: float
            Temperature threshold to use for the test
        :return: np.ndarray
            Array containing the years which first surpassed the threshold
        """
        logic_board = np.argmax(self.data >= threshold, axis=0)
        passing_year = self.time[logic_board]
        return passing_year

    def get_normalised_count_by_year(self, threshold):
        """
        For a given threshold get a list of years in which the threshold was first exceeded in different ensemble
        members along with a count for each year. i.e. 2000 might be the first year in 5 ensemble members. The
        countrs are normalized by the size of the ensemble.

        :param threshold:
        :return:
        """
        passing_year = self.get_exceedance_year(threshold)
        val, count = np.unique(passing_year, return_counts=True)
        count = count[val != 1850]
        count = np.cumsum(count)
        val = val[val != 1850]
        count = count / self.data.shape[1]
        return val, count

    def plot_passing_thresholds(self, filename) -> None:
        """
        The gas gauge plot

        :param filename: str
            Path of the file to which the plot will be written
        :return: None
        """

        plt.figure(figsize=[9, 9])
        for threshold in np.arange(0.5, 1.6, 0.1):

            passing_year = self.get_exceedance_year(threshold)
            val, count = self.get_normalised_count_by_year(threshold)

            cols = [
                '#f7fcfd', '#e5f5f9', '#ccece6', '#99d8c9', '#66c2a4',
                '#41ae76', '#238b45', '#006d2c', '#00441b'
            ]

            plt.text(1979, threshold, f'{threshold:.1f} $\!^\circ\!$C', ha='right', va='center', fontsize=20)
            plt.gca().add_patch(patches.Rectangle(
                (1980, threshold + 0.025), 2024 - 1970, 0.05,
                linewidth=None, edgecolor=None, facecolor='#eeeeee',
                zorder=0
            ))

            for i, p in enumerate([0.0, 0.01, 0.1, 0.33333, 0.5, 0.66666, 0.9, 0.99, 1.0]):
                if max(passing_year) == 1850:
                    continue
                if np.max(count) < p:
                    continue

                threshold_year = val[np.argmax(count >= p)]
                width = 2024 - threshold_year

                plt.gca().add_patch(
                    patches.Rectangle(
                        (threshold_year, threshold - 0.025), width, 0.05,
                        linewidth=None, edgecolor=None, facecolor=cols[i],
                        zorder=p
                    )
                )

                # Plot little black line at the median and label with exact year
                if p == 0.5:
                    plt.plot([threshold_year, threshold_year], [threshold - 0.025, threshold + 0.025],
                             color='black', linewidth=5)
                    plt.text(
                        threshold_year, threshold + 0.05, f'{int(threshold_year)}',
                        ha='center', va='center', fontsize=15, clip_on=False
                    )
                # At the point where it becomes "virtually certain" make a small white mark
                if p == 1:
                    plt.plot([threshold_year, threshold_year], [threshold - 0.025, threshold + 0.025],
                             color='white', linewidth=1)

        # Tidy up
        # Plot x-axis
        for year in [1980, 1990, 2000, 2010, 2020]:
            plt.text(year, 0.5 - 0.025 - 0.02, f'{year}', ha='center', va='top', fontsize=20)
        plt.axis('off')
        plt.gca().set_ylim(0.4, 1.6)
        plt.gca().set_xlim(1980, 2024)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def plot_whole_ensemble(self, filename, alpha=0.01) -> None:
        """
        Plot the whole ensemble as individual midnight blue lines. Alpha is set really low, to give it a smoky vibe.

        :param filename: str
            Path of the file to which the plot will be written
        :return: None
        """
        plt.figure(figsize=[16, 9])
        plt.plot(self.time, self.data, color='midnightblue', alpha=alpha)
        plt.gca().set_ylim(-0.5, 1.75)
        plt.savefig(filename, dpi=300)
        plt.close()

    def plot_joy_division_histogram(self, filename):
        # Calculate some histograms
        plt.figure(figsize=[16, 16])
        for y in range(1850, 2025, 5):
            n_plot_years = 2024 - 1850 + 1
            bins = np.arange(-0.5, 1.77, 0.01)

            h, b = np.histogram(self.data[y - 1850, :], bins=bins)

            b = (b[0:-1] + b[1:]) / 2.
            h_prime = 120 * h / self.n_ensemble + n_plot_years - (y - 1850)
            zero_line = 0 * h + n_plot_years - (y - 1850)

            plt.fill_between(b, zero_line, h_prime, color='white', alpha=1, zorder=y - 1850, clip_on=False)
            plt.plot(b, h_prime, color='black', zorder=y - 1850 + 0.5, clip_on=False)

        plt.gca().set_xlim(-0.5, 1.75)
        plt.gca().set_ylim(0, 175 + 10)
        plt.axis('off')
        plt.gca().set_xlabel(r"Global mean temperature anomaly ($\!^\circ\!$C)")

        for x in [-0.5, 0.0, 0.5, 1.0, 1.5]:
            plt.text(x, -5, f'{x}', ha='center', fontsize=20)

        for y in [1850, 1900, 1950, 2000]:
            plt.text(-0.7, n_plot_years - y + 1850, f'{y}', fontsize=20, va='center')

        plt.savefig(filename, dpi=300)
        plt.close()

    def plot_time_series_with_exceedances(self, filename):

        q1, q2 = self.get_quantile_range(95)
        mn = self.get_ensemble_mean()

        fig, axs = plt.subplots()
        fig.set_size_inches(16, 9)

        axs.fill_between(self.time, q1, q2, color='#d0b2f7', alpha=0.5)
        axs.plot(self.time, mn, color='#6800b8')

        axs.set_xlim(1850, 2024)

        ylims = axs.get_ylim()
        xlims = axs.get_xlim()

        for year in np.arange(1850, 2030, 50):
            plt.text(year, ylims[0] - 0.1, f'{year}', clip_on=False, ha='center', va='center', fontsize=20)

        for threshold in np.arange(0.5, 1.6, 0.5):
            plt.text(1849, threshold, f'{threshold}', clip_on=False, ha='right', va='center', fontsize=20)

            passing_year = self.get_exceedance_year(threshold)
            val, count = self.get_normalised_count_by_year(threshold)

            sides = []

            for p in [0.025, 0.975]:
                if max(passing_year) == 1850:
                    continue
                if np.max(count) < p:
                    continue

                threshold_year = val[np.argmax(count >= p)]
                plt.plot([xlims[0], threshold_year], [threshold, threshold], color='darkgrey')
                sides.append(threshold_year)
                axs.plot([threshold_year, threshold_year], ylims, color='darkgrey')

            if len(sides) == 2:
                axs.plot(sides, [threshold, threshold], linewidth=3, color='#fa5f05')
                plt.gca().add_patch(
                    patches.Rectangle(
                        (sides[0], ylims[0]), sides[1] - sides[0], ylims[1] - ylims[0],
                        linewidth=None, edgecolor=None, facecolor='#f7ceb2',  ##e6a2fc',
                        zorder=0, alpha=0.5
                    )
                )

                median_year = val[np.argmax(count >= 0.5)]

                plt.text(
                    (sides[1] + sides[0]) / 2,
                    ylims[1] + 0.2,
                    f'{threshold}$\!^\circ\!$C',
                    clip_on=False, ha='center', fontsize=20
                )
                plt.text(
                    (sides[1] + sides[0]) / 2,
                    ylims[1] + 0.1,
                    f'{int(median_year)}',
                    clip_on=False, ha='center', fontsize=20
                )
                plt.text(
                    (sides[1] + sides[0]) / 2,
                    ylims[1] + 0.035,
                    f'[{int(sides[0])}-{int(sides[1])}]',
                    clip_on=False, ha='center', fontsize=12
                )

        axs.set_ylim(ylims)
        axs.set_xlim(xlims)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
