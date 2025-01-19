from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

kadow_ensemble_dir = DATA_DIR / 'ManagedData' / 'Data' / 'Kadow_ensemble'

n_ensemble = 200
n_months = 2074
n_years = int(n_months/12)
n_months_whole = n_years * 12

output = np.zeros((n_years, n_ensemble+1))

for member in range(n_ensemble):
    filename = kadow_ensemble_dir / f'20crtaspadzens_tas_mon-gl-72x36_hadcrut5_observation_ens-3_1850-2022_image_{member + 1}.nc'

    print(filename)

    # Open file get area weights
    df = xa.open_dataset(filename)
    weights = np.cos(np.deg2rad(df.tas.latitude))

    # Calculate the area-weighted average, then the annual average
    regional_ts = df.tas.weighted(weights).mean(dim=("latitude", "longitude"))
    regional_ts = regional_ts.data
    regional_ts = regional_ts[0:n_months_whole].astype(np.float16)
    regional_ts = np.mean(regional_ts.reshape(n_years, 12), axis=1)

    # Make a time axis
    time = np.arange(1850,1850+n_years, 1)

    output[:,0] = time[:]
    output[:,member+1] = regional_ts[:]

np.savetxt(kadow_ensemble_dir / "ensemble_time_series.csv", output, delimiter=",")

