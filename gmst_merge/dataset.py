import copy
import numpy as np
import pandas as pd
from random import randrange
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Dataset:

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
    def join(tail, head, join_start_year, join_end_year):

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

    def sample_from_ensemble(self):
        if self.n_ensemble == 1:
            return copy.deepcopy(self)
        else:
            selected_member = randrange(self.n_ensemble - 1)
            out_arr = np.zeros((self.n_time, 2))
            out_arr[:, 0] = self.time[:]
            out_arr[:, 1] = self.data[:, selected_member]
            return Dataset(out_arr)

    def get_start_year(self):
        return int(np.min(self.time))

    def get_end_year(self):
        return int(np.max(self.time))

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

    def lowess_smooth(self):
        """
        Apply a lowess smoother to all ensemble members

        :return:
        """
        lowess = sm.nonparametric.lowess
        smoothed_ensemble = copy.deepcopy(self)
        for i in range(self.n_ensemble):
            z = lowess(self.data[:, i], self.time, frac=1. / 5., return_sorted=False)
            smoothed_ensemble.data[:, i] = z[:]

        return smoothed_ensemble

    def to_csv(self, filename) -> None:
        """
        Write dataset to csv file

        :param filename: str
            File path
        :return: None
        """
        df_full_ensemble = pd.DataFrame(data=self.data.astype(np.float16), index=self.time.astype(int))
        df_full_ensemble.to_csv(filename, sep=',', encoding='utf-8')

    def plot_heat_map(self, filename, normalize=True):

        y1 = self.get_start_year()
        y2 = self.get_end_year()

        bins = np.arange(-0.5, 1.77, 0.01)
        hmap = np.zeros((y2 - y1 + 1, len(bins) - 1))

        for y in range(y1, y2+1):
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
        #hmap[hmap==0] = np.nan
        plt.figure(figsize=[16, 16])
        times = self.time
        plt.pcolormesh(b, times, hmap, cmap='magma', vmin=0, vmax=max_value, shading='nearest')
        plt.gca().tick_params(axis='both', which='major', labelsize=20)
        plt.gca().invert_yaxis()
        plt.savefig(filename, dpi=300)
        plt.close()

    def get_exceedance_year(self, threshold):
        logic_board = np.argmax(self.data >= threshold, axis=0)
        passing_year = self.time[logic_board]
        return passing_year

    def get_normalised_count_by_year(self, threshold):
        passing_year = self.get_exceedance_year(threshold)
        val, count = np.unique(passing_year, return_counts=True)
        count = count[val != 1850]
        count = np.cumsum(count)
        val = val[val != 1850]
        count = count / self.data.shape[1]
        return val, count

    def plot_passing_thresholds(self, filename):

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

    def plot_whole_ensemble(self, filename):
        plt.figure(figsize=[16, 9])
        plt.plot(self.time, self.data, color='midnightblue', alpha=0.01)
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