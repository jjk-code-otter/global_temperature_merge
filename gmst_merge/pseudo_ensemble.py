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
import pandas as pd
import gmst_merge.dataset as ds

if __name__ == '__main__':
    """
    This generates the pseudo ensembles
    """
    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env) / 'ManagedData' / 'Data'

    ensemble_datasets = ["HadCRUT5", "NOAA_ensemble"]

    regular_datasets = [
        "NOAA v5.1", "NOAA v6", "GISTEMP", "CMST3", "COBE-STEMP3", "JRA-3Q"
    ]

    matched_ensembles = {
        "NOAA v5.1": ["NOAA_ensemble"],
        "NOAA v6": ["NOAA_ensemble", "HadCRUT5"],
        "GISTEMP": ["HadCRUT5"],
        "CMST3": ["NOAA_ensemble", "HadCRUT5"],
        "COBE-STEMP3": ["HadCRUT5"],
        "JRA-3Q": ["HadCRUT5"]
    }

    all_perturbations = {}
    all_standardised_perturbations = {}
    for name in ensemble_datasets:
        df = ds.Dataset.read_csv(name, DATA_DIR)
        df.anomalize(1981, 2010)
        df.convert_to_perturbations()
        all_perturbations[name] = df

        df = ds.Dataset.read_csv(name, DATA_DIR)
        df.anomalize(1981, 2010)
        df.convert_to_standardised_perturbations()
        all_standardised_perturbations[name] = df

    all_perturbed_datasets = []
    for name in regular_datasets:
        print(f"Processing {name}")

        # If the uncertainty file exists
        unc_file = DATA_DIR / name / "uncertainty_time_series.csv"
        if unc_file.exists():
            print(f"Using scaled perturbations from {matched_ensembles[name]}")

            ptbs = matched_ensembles[name]
            for ptb2 in ptbs:

                ptb = all_standardised_perturbations[ptb2]

                df = ds.Dataset.read_csv(name, DATA_DIR)
                df.anomalize(1981, 2010)
                y1 = df.get_start_year()
                y2 = df.get_end_year()

                uf = pd.read_csv(unc_file, header=None)
                uf = uf.to_numpy()

                py1 = ptb.get_start_year()
                py2 = ptb.get_end_year()
                if py1 <= y1 and py2 >= y2:
                    all_perturbed_datasets.append(df.make_perturbed_dataset(ptb, scaling=uf))
                else:
                    df = df.select_year_range(py1, py2)
                    uf = uf[py1-y1: py2-y1+1]
                    all_perturbed_datasets.append(df.make_perturbed_dataset(ptb, scaling=uf))
                    print("Length mismatch, subset will be processed")

        else:
            print(f"Using non-scaled perturbations from {matched_ensembles[name]}")

            ptbs = matched_ensembles[name]
            for ptb2 in ptbs:

                ptb = all_perturbations[ptb2]

                df = ds.Dataset.read_csv(name, DATA_DIR)
                df.anomalize(1981, 2010)
                y1 = df.get_start_year()
                y2 = df.get_end_year()

                py1 = ptb.get_start_year()
                py2 = ptb.get_end_year()
                if py1 <= y1 and py2 >= y2:
                    all_perturbed_datasets.append(df.make_perturbed_dataset(ptb))
                else:
                    df = df.select_year_range(py1, py2)
                    all_perturbed_datasets.append(df.make_perturbed_dataset(ptb))
                    print("Length mismatch, subset will be processed")

        print("")

    for df in all_perturbed_datasets:
        directory = DATA_DIR / df.name
        directory.mkdir(exist_ok=True)
        filename = directory / 'ensemble_time_series.csv'
        df.to_csv(filename)
