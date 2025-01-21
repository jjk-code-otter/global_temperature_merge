import copy
import itertools
from pathlib import Path
import os
import random
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randrange
import statsmodels.api as sm
import matplotlib.patches as patches
from typing import List, Tuple


def get_next(j, k, maxj, maxk):
    k += 1
    if k >= maxk:
        k = 0
        j += 1
    return j, k


def split_list(in_lst: list, n_splits: int) -> list:
    """
    Given a list, split it into n_splits random groups

    :param in_lst: list
        List to be split
    :param n_splits: int
        Number of output lists to divide the list into
    :return: list
    """
    new_lst = copy.deepcopy(in_lst)

    # mix the list up
    random.shuffle(new_lst)

    # partition list randomly
    number_of_items = len(new_lst)
    split_points = np.random.choice(number_of_items - 2, n_splits - 1, replace=False) + 1
    split_points.sort()
    result = np.split(new_lst, split_points)

    # convert back to a regular list
    result = [x.tolist() for x in result]

    return result


def pick_one(inarr: list) -> str:
    """
    Given a list of lists, recursively pick from the entries until you find a non-list item and return that

    :param inarr: list
        List containing lists and/or strings
    :return:
    """
    selection = random.choice(inarr)
    if isinstance(selection, list):
        selection = pick_one(selection)
    else:
        return selection
    return selection


def make_random_tree(lst: List[str]) -> list:
    """
    Given a list of datasets, generate a list of lists specifying a hierarchical family tree by repeatedly
    grouping elements.

    :param lst: List[str]
        List of datasets to be
    :return:
    """
    new_list = copy.deepcopy(lst)

    for i in range(4):
        # choose how many breaks to have
        n_items = len(new_list)
        breaks = random.randint(0, n_items - 1)
        # bail if no breaks selected
        if breaks == 0:
            break
        # split list chosen number of times
        new_list = split_list(new_list, breaks)

    return new_list


def choose_start_and_end_year(y1: int, y2: int) -> Tuple[int, int]:
    """
    Given the start and end years of a particular dataset, choose a start year for the overlap

    :param y1: int
        first possible year for overlap
    :param y2: int
        last possible year for overlap
    :return:
        Tuple[int, int]
    """
    trans_start_year = random.randint(y1, y2)
    trans_end_year = trans_start_year + 29
    return trans_start_year, trans_end_year


def anomalize(in_arr: np.ndarray, in_start_year: int, in_end_year: int) -> np.ndarray:
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
    hold = in_arr[0]  # We don't want anomalies of years so hold
    mask = (in_arr[0] >= in_start_year) & (in_arr[0] <= in_end_year)
    out_arr = in_arr - np.mean(in_arr.loc[mask], axis=0)
    out_arr[0] = hold
    return out_arr


def ensemblify(n_meta_ensemble, df, datasets, tails, heads, randomise=True):
    """
    Given all the inputs calculate the meta ensemble with n_meta_ensemble elements

    :param n_meta_ensemble: int
        Size of desired meta ensemble
    :param df:
    :param datasets:
    :param tails:
    :param heads:
    :param randomise:
    :return:
    """
    all_tail_datasets = []
    all_head_datasets = []
    all_start_years = []

    full_ensemble = np.zeros((2024 - 1850 + 1, n_meta_ensemble))

    column_names = []

    for i in range(n_meta_ensemble):

        if randomise:
            hhh = ['HadCRUT5', 'Berkeley Earth', 'NOAA v6', 'GISTEMP_ensemble', 'ERA5', 'JRA-3Q']
            ttt = [
                'DCENT', 'GETQUOCS', 'HadCRUT5', 'Berkeley Earth',
                'NOAA v6', 'Kadow_ensemble', 'Calvert 2024',
                'COBE-STEMP3', 'CMST', 'GloSAT', 'Vaccaro', 'NOAA_ensemble'
            ]

            tails = make_random_tree(ttt)
            heads = make_random_tree(hhh)

        selected_tail = pick_one(tails)
        selected_head = pick_one(heads)

        tail_index = datasets.index(selected_tail)
        head_index = datasets.index(selected_head)

        tail = df[tail_index]
        head = df[head_index]

        cols = tail.columns
        n_members_tail = len(cols) - 1
        if n_members_tail > 0:
            tail_member = randrange(n_members_tail)
        else:
            tail_member = 0

        cols = head.columns
        n_members_head = len(cols) - 1
        if n_members_head > 0:
            head_member = randrange(n_members_head)
        else:
            head_member = 0

        tail = tail[[0, tail_member + 1]]
        head = head[[0, head_member + 1]]

        # Choose start and end year
        trans_start_year, trans_end_year = choose_start_and_end_year(np.min(head[0]), 1981)

        tail = anomalize(tail, trans_start_year, trans_end_year)
        head = anomalize(head, trans_start_year, trans_end_year)

        merged = pd.merge(tail, head, how='outer', on=0)

        merged = merged.to_numpy()

        select_tail = merged[:, 0] < (trans_start_year + trans_end_year) / 2
        select_head = merged[:, 0] >= (trans_start_year + trans_end_year) / 2

        merged[select_head, 1] = merged[select_head, 2]

        select_climatology = (merged[:, 0] >= 1981) & (merged[:, 0] <= 2010)
        full_ensemble[:, i] = merged[:, 1] - np.mean(merged[select_climatology, 1])

        all_tail_datasets.append(selected_tail)
        all_head_datasets.append(selected_head)
        all_start_years.append(trans_start_year)

        column_names.append(f"{selected_tail} {selected_head} {trans_start_year}")

    stats = pd.DataFrame(
        {'tails': all_tail_datasets, 'heads': all_head_datasets, 'years': all_start_years}
    )

    time = merged[:, 0]

    return stats, time, full_ensemble, column_names


def change_baseline(time, ensemble, y1, y2):
    select_baseline = (time >= y1) & (time <= y2)
    rebaselined_ensemble = copy.deepcopy(ensemble)
    for i in range(ensemble.shape[1]):
        rebaselined_ensemble[:, i] = ensemble[:, i] - np.mean(ensemble[select_baseline, i])
    return rebaselined_ensemble


def lowess_smooth(time, ensemble):
    lowess = sm.nonparametric.lowess
    smoothed_ensemble = copy.deepcopy(ensemble)
    for i in range(ensemble.shape[1]):
        z = lowess(ensemble[:, i], time, frac=1. / 5., return_sorted=False)
        smoothed_ensemble[:, i] = z[:]
    return smoothed_ensemble


def get_exceedance_year(ensemble_array, threshold):
    logic_board = np.argmax(ensemble_array >= threshold, axis=0)
    passing_year = logic_board + 1850
    return passing_year


def get_normalised_count_by_year(ensemble_array, passing_year):
    val, count = np.unique(passing_year, return_counts=True)
    count = count[val != 1850]
    count = np.cumsum(count)
    val = val[val != 1850]
    count = count / ensemble_array.shape[1]
    return val, count


def summarise(ensemble):
    return np.mean(ensemble, axis=1), np.std(ensemble, axis=1), np.quantile(ensemble, 0.025, axis=1), np.quantile(
        ensemble, 0.975, axis=1)


# Plotting code
def plot_input_ensemble(df, in_start_year, in_end_year, img_filename):
    fig, axs = plt.subplots(4, 4)
    fig.set_size_inches(16, 16)

    k = 0
    l = 0

    for i, d in enumerate(df):

        k, l = get_next(k, l, 4, 4)

        cols = d.columns
        n_members = len(cols) - 1

        # Calculate anomalies
        hold = d[0]  # We don't want anomalies of years so hold
        mask = (d[0] >= in_start_year) & (d[0] <= in_end_year)
        d = d - np.mean(d.loc[mask], axis=0)
        d[0] = hold

        for j in range(n_members):
            if j == 0:
                axs[k][l].plot(d[0], d[j + 1], color='red', label=datasets[i], zorder=10)
            else:
                axs[k][l].plot(d[0], d[j + 1], color='red', zorder=10)

            # plot all datasets in grey in all panels
            for m, n in itertools.product(range(4), range(4)):
                axs[m][n].plot(d[0], d[j + 1], color='grey', alpha=0.2, zorder=1)

        # axs[k][l].legend()
        axs[k][l].text(1856, 0.8, datasets[i], fontsize=13)
        axs[k][l].set_xlim(1850, 2024)
        axs[k][l].set_ylim(-1.5, 1.0)

    for m, n in itertools.product(range(4), range(4)):
        axs[m][n].spines['right'].set_visible(False)
        axs[m][n].spines['top'].set_visible(False)

    plt.savefig(img_filename, dpi=300)
    plt.close()


def make_joy_division_histogram(ensemble_array, filename):
    # Calculate some histograms
    plt.figure(figsize=[16, 16])
    for y in range(1850, 2025, 5):
        n_plot_years = 2024 - 1850 + 1
        bins = np.arange(-0.5, 1.77, 0.01)

        h, b = np.histogram(ensemble_array[y - 1850, :], bins=bins)

        b = (b[0:-1] + b[1:]) / 2.
        h_prime = 120 * h / n_meta_ensemble + n_plot_years - (y - 1850)
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


def make_heat_map(ensemble_array, filename):
    bins = np.arange(-0.5, 1.77, 0.01)
    hmap = np.zeros((2024 - 1850 + 1, len(bins) - 1))

    for y in range(1850, 2025):
        n_plot_years = 2024 - 1850 + 1
        bins = np.arange(-0.5, 1.77, 0.01)
        h, b = np.histogram(ensemble_array[y - 1850, :], bins=bins)
        b = (b[0:-1] + b[1:]) / 2.
        h_prime = h / n_meta_ensemble
        hmap[y - 1850, :] = h_prime / np.max(h_prime)

    max_value = np.max(hmap)
    #    hmap[hmap==0] = np.nan
    plt.figure(figsize=[16, 16])
    times = np.arange(1850, 2025)
    plt.pcolormesh(b, times, hmap, cmap='magma', vmin=0, vmax=max_value, shading='nearest')
    plt.gca().tick_params(axis='both', which='major', labelsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(filename, dpi=300)
    plt.close()


def passing_thresholds(ensemble_array, filename):
    plt.figure(figsize=[9, 9])
    for threshold in np.arange(0.5, 1.6, 0.1):
        passing_year = get_exceedance_year(ensemble_array, threshold)
        val, count = get_normalised_count_by_year(ensemble_array, passing_year)

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

            # Plot little line at the median and label with exact year
            if p == 0.5:
                plt.plot([threshold_year, threshold_year], [threshold - 0.025, threshold + 0.025],
                         color='black', linewidth=5)
                plt.text(
                    threshold_year, threshold + 0.05, f'{threshold_year}',
                    ha='center', va='center', fontsize=15, clip_on=False
                )
            if p == 1:
                plt.plot([threshold_year, threshold_year], [threshold - 0.025, threshold + 0.025],
                         color='white', linewidth=1)

    # Tidy up
    for year in [1980, 1990, 2000, 2010, 2020]:
        plt.text(year, 0.5 - 0.025 - 0.02, f'{year}', ha='center', va='top', fontsize=20)
    plt.axis('off')
    plt.gca().set_ylim(0.4, 1.6)
    plt.gca().set_xlim(1980, 2024)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_whole_ensemble(filename, time, inarr):
    plt.figure(figsize=[16, 9])
    for i in range(inarr.shape[1]):
        plt.plot(time, inarr[:, i], color='midnightblue', alpha=0.01)
    plt.gca().set_ylim(-0.5, 1.75)
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_simple_summary(filename, time, full_ensemble_mean, full_ensemble_stdev, cover_factor=1.96, ylim=[-1.25, 1.0]):
    fig, axs = plt.subplots(2, sharex=True)
    fig.set_size_inches(16, 9)

    colours = ['red', 'blue']

    for i, mn in enumerate(full_ensemble_mean):
        axs[0].plot(time, mn, color='black')
        axs[0].fill_between(
            time,
            mn + cover_factor * full_ensemble_stdev[i],
            mn - cover_factor * full_ensemble_stdev[i],
            color=colours[i], alpha=0.2
        )
    axs[0].set_title('Mean and 95% range', loc='left')
    axs[0].set_ylim(ylim[0], ylim[1])

    for i, sd in enumerate(full_ensemble_stdev):
        axs[1].plot(time, cover_factor * sd, color=colours[i])
    axs[1].set_title(f'Standard deviation * {cover_factor}', loc='left')
    axs[1].set_ylim(0, 0.3)

    plt.savefig(filename, dpi=300)
    plt.close()

    return


# Output functions, csv writers etc
def ensemble_to_csv(filename, time, full_ensemble, column_names):
    df_full_ensemble = pd.DataFrame(data=full_ensemble.astype(np.float16), index=time.astype(int),
                                    columns=column_names)
    df_full_ensemble.to_csv(filename, sep=',', encoding='utf-8')


def all_to_csv(
        filename,
        time,
        full_ensemble_mean, full_ensemble_stdev, f025, f975,
        full_ensemble_pre_mean, full_ensemble_pre_stdev, fp025, fp975,
        full_ensemble_lowess_mean, full_ensemble_lowess_stdev, fl025, fl975
):
    df_ensemble_summary = pd.DataFrame(
        {
            'year': time.astype(int),
            'mean': full_ensemble_mean.astype(np.float16), 'stdev': full_ensemble_stdev.astype(np.float16),
            'q025': f025.astype(np.float16), 'q975': f975.astype(np.float16),
            'mean preindustrial': full_ensemble_pre_mean.astype(np.float16),
            'stdev preindustrial': full_ensemble_pre_stdev.astype(np.float16),
            'q025 preindustrial': fp025.astype(np.float16), 'q975 preindustrial': fp975.astype(np.float16),
            'mean lowess': full_ensemble_lowess_mean.astype(np.float16),
            'stdev lowess': full_ensemble_lowess_stdev.astype(np.float16),
            'q025 lowess': fl025.astype(np.float16), 'q975 lowess': fl975.astype(np.float16),
        }
    )
    df_ensemble_summary.to_csv(filename, sep=',', encoding='utf-8')


if __name__ == '__main__':
    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env) / 'ManagedData' / 'Data'

    start_year = 1981
    end_year = 2010
    n_meta_ensemble = 10000
    cover_factor = 1.96  # 1.645
    plot_full_input_ensemble = False

    # Each dataset directory contains a file with this name which has n + 1 columns and m rows where n is the
    # number of ensemble members and m is the number of years. The first column is the year column
    filestub = 'ensemble_time_series.csv'

    datasets = [
        'DCENT', 'GETQUOCS', 'HadCRUT5', 'Berkeley Earth',
        'NOAA v6', 'Kadow_ensemble', 'Calvert 2024',
        'COBE-STEMP3', 'CMST', 'GISTEMP_ensemble', 'ERA5',
        'JRA-3Q', 'GloSAT', 'Vaccaro', 'NOAA_ensemble'
    ]

    hhh = ['HadCRUT5', 'Berkeley Earth', 'NOAA v6', 'GISTEMP_ensemble', 'ERA5', 'JRA-3Q']
    ttt = [
        'DCENT', 'GETQUOCS', 'HadCRUT5', 'Berkeley Earth',
        'NOAA v6', 'Kadow_ensemble', 'Calvert 2024',
        'COBE-STEMP3', 'CMST', 'GloSAT', 'Vaccaro', 'NOAA_ensemble'
    ]

    filenames = [DATA_DIR / x / filestub for x in datasets]

    colours = [
        'grey', 'green', 'black', 'blue',
        'orange', 'goldenrod', 'goldenrod',
        'dodgerblue', 'yellow', 'purple',
        'dodgerblue', 'violet', 'violet',
        'indianred', 'lightseagreen', 'fuchsia'
    ]

    for hierarchy_variable in ['ur_ensembles_only', 'ur', 'random', 'sst', 'lsat', 'interp', 'sst_ensembles_only',
                               'lsat_ensembles_only', 'equal']:

        print(f"Processing hierarchy based on {hierarchy_variable}")

        if hierarchy_variable == 'random':
            tails = make_random_tree(ttt)
            heads = make_random_tree(hhh)
            random_switch = True
        else:
            with open(f'FamilyTrees/hierarchy_{hierarchy_variable}.json') as f:
                hierarch = json.load(f)

            master = hierarch['master']
            tails = hierarch['tails']
            heads = hierarch['heads']
            random_switch = False

        version = f'v4_{hierarchy_variable}'

        df = []

        for file in filenames:
            df.append(pd.read_csv(file, header=None))

        if hierarchy_variable == 'ur_ensembles_only' and plot_full_input_ensemble:
            plot_input_ensemble(
                df, start_year, end_year,
                f'Figures/worm_ensemble_inputs_{version}.png'
            )

        stats, time, full_ensemble, column_names = ensemblify(n_meta_ensemble, df, datasets, tails, heads,
                                                              randomise=random_switch)

        full_ensemble_pre = change_baseline(time, full_ensemble, 1850, 1900)
        full_ensemble_lowess = lowess_smooth(time, full_ensemble_pre)

        plot_whole_ensemble(f'Figures/worm_ensemble_{version}.png', time, full_ensemble_lowess)

        passing_thresholds(full_ensemble_lowess, f'Figures/passing_threshold_{version}.png')
        make_heat_map(full_ensemble_pre, f'Figures/hmap_{version}.png')
        make_heat_map(full_ensemble_lowess, f'Figures/hmap_lowess_{version}.png')
        make_joy_division_histogram(full_ensemble_pre, f'Figures/joy_division_{version}.png')
        make_joy_division_histogram(full_ensemble_lowess, f'Figures/joy_division_lowess_{version}.png')

        full_ensemble_mean, full_ensemble_stdev, f025, f975 = summarise(full_ensemble)
        full_ensemble_pre_mean, full_ensemble_pre_stdev, fp025, fp975 = summarise(full_ensemble_pre)
        full_ensemble_lowess_mean, full_ensemble_lowess_stdev, fl025, fl975 = summarise(full_ensemble_lowess)

        ensemble_to_csv(f'Output/ensemble_summary_{hierarchy_variable}.csv', time, full_ensemble, column_names)
        all_to_csv(
            f'Output/full_summary_{hierarchy_variable}.csv',
            time,
            full_ensemble_mean, full_ensemble_stdev, f025, f975,
            full_ensemble_pre_mean, full_ensemble_pre_stdev, fp025, fp975,
            full_ensemble_lowess_mean, full_ensemble_lowess_stdev, fl025, fl975
        )

        plot_simple_summary(
            f'Figures/worm_summary_{version}.png',
            time,
            [full_ensemble_mean],
            [full_ensemble_stdev]
        )

        plot_simple_summary(
            f'Figures/worm_summary_preind_{version}.png',
            time,
            [full_ensemble_pre_mean],
            [full_ensemble_pre_stdev],
            ylim=[-0.5, 1.75]
        )

        plot_simple_summary(
            f'Figures/worm_summary_lowess_{version}.png',
            time,
            [full_ensemble_lowess_mean, full_ensemble_pre_mean],
            [full_ensemble_lowess_stdev, full_ensemble_pre_stdev],
            ylim=[-0.5, 1.75]
        )

        selected_year = (time >= 2001) & (time <= 2020)
        early_average = np.mean(full_ensemble_pre[selected_year, :], axis=0)

        print("Warming from 1850-1900 to 2001-2020 (IPCC metric)")
        print(
            f"Mean {np.mean(early_average):.2f} and 1.645 * stdev {1.645 * np.std(early_average):.2f} and 90% range {np.quantile(early_average, 0.05):.2f}-{np.quantile(early_average, 0.95):.2f}")

        selected_year_a = (time >= 1880) & (time <= 1899)
        selected_year_b = (time >= 1995) & (time <= 2014)
        early_average = np.mean(full_ensemble_pre[selected_year_a, :], axis=0)
        late_average = np.mean(full_ensemble_pre[selected_year_b, :], axis=0)
        difference = late_average - early_average

        print("Warming from 1880-1899 to 1995-2014 (C and G metric)")
        print(
            f"Mean {np.mean(difference):.3f} and 1.96 * stdev {1.96 * np.std(difference):.3f} and 95% range {np.quantile(difference, 0.025):.3f}-{np.quantile(difference, 0.975):.3f}")

        print("Prob that lowess exceeded 1.5, 1.4 and 1.3C in 2024")
        print(100 * np.count_nonzero(full_ensemble_lowess[2024 - 1850, :] > 1.5) / n_meta_ensemble)
        print(100 * np.count_nonzero(full_ensemble_lowess[2024 - 1850, :] > 1.4) / n_meta_ensemble)
        print(100 * np.count_nonzero(full_ensemble_lowess[2024 - 1850, :] > 1.3) / n_meta_ensemble)

        print("Prob that 2024 exceeded 1.5C, 2023 exceeded 1.5 and that 2024 was the first year to do so")
        print(100 * np.count_nonzero(full_ensemble_pre[2024 - 1850, :] > 1.5) / n_meta_ensemble)
        print(100 * np.count_nonzero(full_ensemble_pre[2023 - 1850, :] > 1.5) / n_meta_ensemble)
        print(100 * np.count_nonzero(
            (full_ensemble_pre[2024 - 1850, :] > 1.5) & (full_ensemble_pre[2023 - 1850, :] < 1.5)) / n_meta_ensemble)

        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(16, 9)
        stats.hist(ax=axs[0][0], column='years', bins=np.arange(1849, 1982, 1) + 0.5)
        stats['heads'].value_counts().plot(ax=axs[0][1], kind='bar')
        stats['tails'].value_counts().plot(ax=axs[1][0], kind='bar')
        axs[1][1].axis('off')
        plt.savefig(f'Figures/histo_summary_{hierarchy_variable}.png', dpi=300, bbox_inches='tight')
        plt.close()
