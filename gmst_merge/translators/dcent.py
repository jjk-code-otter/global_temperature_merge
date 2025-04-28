from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'DCENT'

n_years = 2024-1850+1
n_months = 12 * n_years
n_ensemble = 200

output = np.zeros((n_years, n_ensemble+1))

for i in range(1, n_ensemble+1):
    # filename =  dcent_dir / f'DCENT_ensemble_1850_2023_member_{i:03d}.nc'
    filename = data_file_dir / f'DCENT_ensemble_reso_5_1850_2025_member_{i:03d}.nc'
    print(filename)

    # Open file get area weights
    df = xa.open_dataset(filename)

    weights = np.cos(np.deg2rad(df.temperature.lat))

    # Calculate the area-weighted average, then the annual average
    regional_ts = df.temperature.weighted(weights).mean(dim=("lat", "lon"))
    regional_ts = regional_ts.data
    regional_ts = np.mean(regional_ts.reshape(int(n_months/12), 12), axis=1)
    output[:,i] = regional_ts[:]

# Transpose array and add time axis for writing
time = np.arange(1850,1850+n_years, 1)
output[:,0] = time[:]

np.savetxt(data_file_dir / "ensemble_time_series.csv", output, fmt='%.4f', delimiter=",")

