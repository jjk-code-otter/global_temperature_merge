import copy
from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def make_perturbations(ensemble):
    """
    Given an ensemble array (ntime, n_ensemble_members) then calculate the mean of the ensemble members and
    subtract that from each ensemble member

    Args:
        ensemble: ndarray

    Returns:

    """
    perturbations = copy.deepcopy(ensemble)
    n_time = perturbations.shape[0]
    mean_series = np.mean(ensemble[:, 1:], axis=1)
    perturbations[:, 1:] = ensemble[:, 1:] - np.reshape(mean_series, (n_time, 1))
    return perturbations


def perturb_dataset(dataset, perturbations):
    fyr = int(dataset[0, 0])
    lyr = int(dataset[-1, 0])
    p_fyr = int(perturbations[0, 0])

    n_time = dataset.shape[0]
    n_ensemble_members = perturbations.shape[1]
    out_dataset = np.zeros((n_time, n_ensemble_members))

    out_dataset[:, 0] = dataset[:, 0]
    out_dataset[:, 1:] = np.reshape(dataset[:, 1], (n_time, 1)) + perturbations[fyr - p_fyr:lyr - p_fyr + 1, 1:]

    return out_dataset


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

    ensemble_datasets = [
        'DCENT', 'GETQUOCS', 'HadCRUT5', 'Kadow_ensemble', 'Calvert 2024',
        'GISTEMP_ensemble', 'GloSAT', 'Vaccaro', 'NOAA_ensemble'
    ]
    ensemble_datasets = ['HadCRUT5']

    non_ensemble_datasets = ['Berkeley Earth', 'NOAA v6', 'COBE-STEMP3', 'CMST', 'ERA5', 'JRA-3Q']

    filenames = [DATA_DIR / x / filestub for x in ensemble_datasets]
    ensemble_df = []
    for file in filenames:
        ensemble_df.append(pd.read_csv(file, header=None).to_numpy())

    filenames = [DATA_DIR / x / filestub for x in non_ensemble_datasets]
    non_ensemble_df = []
    for file in filenames:
        non_ensemble_df.append(pd.read_csv(file, header=None).to_numpy())

    all_perturbations = []
    for df in ensemble_df:
        all_perturbations.append(make_perturbations(df))

    all_perturbed = []
    for df in non_ensemble_df:
        all_perturbed.append(perturb_dataset(df, all_perturbations[0]))

    for p in all_perturbed:
        plt.plot(p[:, 0], p[:, 1:])
    plt.show()
    plt.close()
