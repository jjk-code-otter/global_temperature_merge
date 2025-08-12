from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'DCENT_MLE_v1p1'
filename = data_file_dir / 'DCENT_MLE_v1.1_timeseries_annual_anomalies_ensemble.nc'

df = xa.open_dataset(filename)

ntime = df.surface_temperature_anomaly.shape[1]
nensemble = df.surface_temperature_anomaly.shape[0]

output = np.zeros((ntime, nensemble + 1))
output[:, 1:] = np.transpose(df.surface_temperature_anomaly.values[:, :])
output[:, 0] = np.arange(1850, 1850 + ntime, 1)
output = output.astype(np.float16)

np.savetxt(data_file_dir / "ensemble_time_series.csv", output, fmt='%.4f', delimiter=",")
