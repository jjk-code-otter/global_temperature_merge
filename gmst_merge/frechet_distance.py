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
    cov1 = np.cov(data1, ddof=0)
    cov2 = np.cov(data2, ddof=0)

    # use eigendecomposition to calculate square root to deal with cases with nonpositive eigenvalues
    # eigenvalues can be slightly negative due to numeric error
    eigenvalues, eigenvectors = sp.linalg.eigh(cov1)
    sqrt_cov1 = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    eigenvalues, eigenvectors = sp.linalg.eigh(sqrt_cov1 @ cov2 @ sqrt_cov1)

    A = np.sum(np.square(mean1 - mean2))
    B = np.trace(cov1 + cov2 - 2 * eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T)

    return A + B


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

    datasets = {name: ds.Dataset.read_csv_from_file(fname, name) for (name, fname) in tab.items()}

    # Add in the rescaled ensemble
    filename = out_path / 'final_small_ensemble' / 'sst_pseudo.csv'
    rescaled_ensemble = ds.Dataset.read_csv_from_file(filename, "small_ensemble")
    rescaled_ensemble.rescale_ensemble(datasets["large_ensemble"])
    datasets["rescaled_ensemble"] = rescaled_ensemble

    # calculate Frechet distances relative to large ensemble for all ensembles
    frechet_distances = {
        name: Frechet_distance(ds.data, datasets["large_ensemble"].data) for name, ds in datasets.items()
    }

    # Then write the results out
    with open(Path("Output/Frechet.txt"), 'w') as file:
        for name, distance in frechet_distances.items():
            file.write(f"{name}: {distance}\n")
