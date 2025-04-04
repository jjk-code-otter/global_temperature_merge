from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

hadcrut5_dir = DATA_DIR / 'ManagedData' / 'Data' / 'CMST'
filename = hadcrut5_dir / 'China-MST2.0-Imax.nc'

df= xa.open_dataset(filename)
weights = np.cos(np.deg2rad(df.lat))
area_average = df.weighted(weights).mean(dim=("lat", "lon"))

anomalies = area_average.T2mSST_anomaly.data

ntimes = len(anomalies)
nyears = int(ntimes / 12)

output = np.zeros((nyears, 2))

time = np.arange(1850,1850+nyears, 1)

output[:, 0] = time
output[:, 1] = np.mean(anomalies.reshape(nyears, 12), axis=1)

np.savetxt(hadcrut5_dir / "ensemble_time_series.csv", output, fmt='%.4f', delimiter=",")

# Now the uncertainties digitised from the paper Figure 8b
# https://essd.copernicus.org/articles/14/1677/2022/#section3
df = pd.read_csv(hadcrut5_dir / 'plot-data.csv')

nyears = len(df.time)

output = np.zeros((nyears, 2))

output[:, 0] = df.time
output[:, 1] = df.uncertainty

np.savetxt(hadcrut5_dir / "uncertainty_time_series.csv", output, fmt='%.4f', delimiter=",")
