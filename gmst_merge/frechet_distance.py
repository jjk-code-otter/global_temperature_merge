#  Code for Frechet distance performance tests of different ensemble generation methods.
#  by Bruce T. T. Calvert

from pathlib import Path
import dataset as ds
import numpy as np
import scipy as sp


def Frechet_distance(data1, data2):
    """
    Returns Frechet distance given two numpy arrays

    :param data1: numpy array
    :param data2: numpy array
    :return:
    """
    if (
            data1.shape[0] != data2.shape[0] or
            data1.shape[1] < 2 or
            data2.shape[1] < 2 or
            not isinstance(data1, np.ndarray) or
            not isinstance(data2, np.ndarray)
    ):
        raise RuntimeError('Invalid inputs for calculating Frechet distance')

    mean1 = np.mean(data1, axis=1)
    mean2 = np.mean(data2, axis=1)
    cov1 = np.cov(data1, ddof=1)
    cov2 = np.cov(data2, ddof=1)

    return np.sum(np.square(mean1 - mean2)) + np.trace(cov1 + cov2 - 2 * sp.linalg.sqrtm(np.matmul(cov1, cov2)))


if __name__ == '__main__':

    out_path = Path('Output')

    tab = {
        "large_ensemble": out_path / 'basic' / 'sst_pseudo.csv',
        "medium_ensemble": out_path / 'medium_ensemble' / 'sst_pseudo.csv',
        "small_ensemble": out_path / 'final_small_ensemble' / 'sst_pseudo.csv',
        "clustered_ensemble": out_path / 'final_clustered' / 'sst_pseudo.csv',
        "thinned_ensemble": out_path / 'final_thinned' / 'sst_pseudo.csv',
        "balanced_ensemble": out_path / 'balanced_ensemble' / 'sst_pseudo.csv',
    }

    datasets = { name: ds.Dataset.read_csv_from_file(fname, name) for (name, fname) in tab.items()}

    # Add in the rescaled ensemble
    filename = out_path / 'final_small_ensemble' / 'sst_pseudo.csv'
    rescaled_ensemble = ds.Dataset.read_csv_from_file(filename, "small_ensemble")
    rescaled_ensemble.rescale_ensemble(datasets["large_ensemble"])
    datasets["rescaled_ensemble"] = rescaled_ensemble

    # calculate Frechet distances relative to large ensemble
    frechet_distances = {
        name: Frechet_distance(ds.data, datasets["large_ensemble"].data) for name, ds in datasets.items()
    }

    with open(Path("Output/Frechet.txt"), 'w') as file:
        for name, distance in frechet_distances.items():
            file.write(f"{name}: {distance}\n")