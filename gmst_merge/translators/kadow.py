from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt



data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'Kadow'
filename = data_file_dir / 'HadCRUT5.anomalies.Kadow_et_al_2020_20crAI-infilled.ensemble_mean_global_mean_185001-202312.nc'

df = xa.open_dataset(filename)

ntimes = df.tas.data.shape[0]
nyears = int(ntimes/12)

output = np.zeros((nyears, 2))

years = df.time.dt.year.data
months = df.time.dt.month.data
anomalies = np.reshape(df.tas.data, (ntimes))

output[:,0] = np.arange(1850,2024,1)
output[:,1] = np.mean(anomalies.reshape(nyears, 12), axis=1)

np.savetxt(data_file_dir / "ensemble_time_series.csv", output, delimiter=",")

