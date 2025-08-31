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

import os
from pathlib import Path
import pandas as pd
import gmst_merge.dataset as ds


def make_perturbations(ensemble_datasets, clim1, clim2, data_dir):
    all_perturbations = {}
    all_standardised_perturbations = {}

    for name in ensemble_datasets:
        df = ds.Dataset.read_csv(name, data_dir)
        df.anomalize(clim1, clim2)
        df.convert_to_perturbations()
        all_perturbations[name] = df

        df = ds.Dataset.read_csv(name, data_dir)
        df.anomalize(clim1, clim2)
        df.convert_to_standardised_perturbations()
        all_standardised_perturbations[name] = df

    return all_perturbations, all_standardised_perturbations


def apply_perturbations(name, data_dir, unc_file, ptb, clim1, clim2):
    df = ds.Dataset.read_csv(name, data_dir)

    df.anomalize(clim1, clim2)
    y1 = df.get_start_year()
    y2 = df.get_end_year()

    uf = None
    if unc_file is not None:
        uf = pd.read_csv(unc_file, header=None)
        uf = uf.to_numpy()

    py1 = ptb.get_start_year()
    py2 = ptb.get_end_year()

    if py1 <= y1 and py2 >= y2:
        print(f"Lengths of dataset and perturbations match.")
        perturbed_dataset = df.make_perturbed_dataset(ptb, scaling=uf)
    else:
        print(f"Length mismatch, dataset {y1}-{y2}, perturbations {py1}-{py2} subset will be processed.")
        df = df.select_year_range(py1, py2)
        if uf is not None:
            uf = uf[py1 - y1: py2 - y1 + 1]
        perturbed_dataset = df.make_perturbed_dataset(ptb, scaling=uf)


    return perturbed_dataset


if __name__ == '__main__':
    """
    This generates the pseudo ensembles
    """
    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env) / 'ManagedData' / 'Data'

    ensemble_datasets = ["HadCRUT5", "NOAA_ensemble", "ERA5 ensemble"]

    regular_datasets = [
        "NOAA v5.1", "NOAA v6", "GISTEMP", "CMST3", "COBE-STEMP3", "JRA-3Q"
    ]

    matched_ensembles = {
        "NOAA v5.1": ["NOAA_ensemble"],
        "NOAA v6": ["NOAA_ensemble", "HadCRUT5"],
        "GISTEMP": ["HadCRUT5"],
        "CMST3": ["NOAA_ensemble", "HadCRUT5"],
        "COBE-STEMP3": ["HadCRUT5"],
        "JRA-3Q": ["ERA5 ensemble"]
    }

    baselines = {
        "NOAA v5.1": [1971, 2000],  # see https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v6/access/timeseries/00_Readme_timeseries.txt
        "NOAA v6": [1971, 2000],  # see https://www.ncei.noaa.gov/data/noaa-global-surface-temperature/v6/access/timeseries/00_Readme_timeseries.txt
        "GISTEMP": [1951, 1980],  # See https://data.giss.nasa.gov/gistemp/
        "CMST3": [1961, 1990],  # See http://www.gwpu.net/en/h-col-103.html
        "COBE-STEMP3": [1961, 1990],  # Inferred from input file
        "JRA-3Q": [1981, 2010]  # Doesn't matter in this case because uncertainty is taken neat from ERA5
    }

    all_perturbed_datasets = []

    for name in regular_datasets:
        print(f"Processing {name} - with climatology {baselines[name][0]}-{baselines[name][1]}")

        clim1 = baselines[name][0]
        clim2 = baselines[name][1]

        all_perturbations, all_standardised_perturbations = make_perturbations(
            ensemble_datasets, clim1, clim2, DATA_DIR
        )

        unc_file = DATA_DIR / name / "uncertainty_time_series.csv"
        if not unc_file.exists():
            print(f"Using non-scaled perturbations from {matched_ensembles[name]}")
            unc_file = None
            all_selected_perturbations = all_perturbations
        else:
            print(f"Using scaled perturbations from {matched_ensembles[name]}")
            all_selected_perturbations = all_standardised_perturbations

        ptbs = matched_ensembles[name]
        for ptb2 in ptbs:
            print(ptb2)
            ptb = all_selected_perturbations[ptb2]
            perturbed_dataset = apply_perturbations(name, DATA_DIR, unc_file, ptb, clim1, clim2)
            all_perturbed_datasets.append(perturbed_dataset)

        print("")

    for df in all_perturbed_datasets:
        directory = DATA_DIR / df.name
        directory.mkdir(exist_ok=True)
        filename = directory / 'ensemble_time_series.csv'
        df.to_csv(filename)
