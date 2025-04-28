import itertools
from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'Kadow_new_ensemble'

n_ensemble = 200
n_subensemble = 5
n_years = 2024 - 1850 + 1
n_months = 12 * n_years

output = np.zeros((n_years, n_ensemble * n_subensemble + 1))

counter = 0
for submember, member in itertools.product(range(n_subensemble), range(n_ensemble)):
    filename = data_file_dir / f'Kadow_et_al_2025_AI-Infilled_HadCRUT.5.0.2.0.anomalies.annualmean.globalmean.185001-202412_AIens-{submember + 1}_H5ens-{member + 1}.nc'

    # Open file get area weights
    df = xa.open_dataset(filename)

    # Make a time axis
    time = np.arange(1850, 1850 + n_years, 1)

    output[:, 0] = time[:]
    output[:, counter + 1] = df.tas_mean.data[:, 0, 0]

    counter += 1

output = output.astype(np.float16)

np.savetxt(data_file_dir / "ensemble_time_series.csv", output, fmt='%.4f', delimiter=",")
