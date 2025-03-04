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

import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import gmst_merge.family_tree as ft
import gmst_merge.dataset as ds
import gmst_merge.metaensemblefactory as mef

if __name__ == '__main__':
    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env) / 'ManagedData' / 'Data'

    all_ensemble_datasets = [
        "GloSAT", "HadCRUT5", "GETQUOCS",
        "Calvert 2024", "Vaccaro", "GISTEMP_ensemble",
        "Kadow_ensemble", "NOAA_ensemble", "DCENT"
    ]

    ensemble_datasets = ["HadCRUT5"]

    regular_datasets = [
        "Berkeley Earth", "NOAA v6", "CMST", "COBE-STEMP3", "ERA5", "JRA-3Q"
    ]

    all_perturbations = []
    all_standardised_perturbations = []
    for name in ensemble_datasets:
        df = ds.Dataset.read_csv(name, DATA_DIR)
        df.anomalize(1981, 2010)
        df.convert_to_perturbations()
        all_perturbations.append(df)
        plt.plot(df.time, df.data, alpha=0.1)

        df = ds.Dataset.read_csv(name, DATA_DIR)
        df.anomalize(1981, 2010)
        df.convert_to_standardised_perturbations()
        all_standardised_perturbations.append(df)

    plt.show()
    plt.close()

    all_perturbed_datasets = []
    for name in regular_datasets:
        df = ds.Dataset.read_csv(name, DATA_DIR)
        df.anomalize(1981, 2010)
        y1 = df.get_start_year()
        y2 = df.get_end_year()

        # If the uncertainty file exists
        unc_file = DATA_DIR / name / "uncertainty_time_series.csv"
        if unc_file.exists():
            uf = pd.read_csv(unc_file, header=None)
            uf = uf.to_numpy()
            matching_perturbations = []
            for ptb in all_standardised_perturbations:
                py1 = ptb.get_start_year()
                py2 = ptb.get_end_year()
                if py1 <= y1 and py2 >= y2:
                    all_perturbed_datasets.append(df.make_perturbed_dataset(ptb, scaling=uf))
        else:
            matching_perturbations = []
            for ptb in all_perturbations:
                py1 = ptb.get_start_year()
                py2 = ptb.get_end_year()
                if py1 <= y1 and py2 >= y2:
                    all_perturbed_datasets.append(df.make_perturbed_dataset(ptb))

    for df in all_perturbed_datasets:
        plt.plot(df.time, df.data, alpha=0.01, linewidth=2)
        directory = DATA_DIR / df.name
        directory.mkdir(exist_ok=True)
        filename = directory / 'ensemble_time_series.csv'
        df.to_csv(filename)

    plt.show()
    plt.close()
