from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

hadcrut5_dir = DATA_DIR / 'ManagedData' / 'Data' / 'DCENT_MLE'
filename = hadcrut5_dir / 'DCENT_MLE_v1.0_timeseries_annual_anomalies_ensemble.nc'

df = xa.open_dataset(filename)

output = np.zeros((174,201))
output[:,1:] = np.transpose(df.surface_temperature_anomaly.values[:,:])
output[:,0] = np.arange(1850, 1850+174, 1)
output = output.astype(np.float16)

np.savetxt(hadcrut5_dir / "ensemble_time_series.csv", output, fmt='%.4f', delimiter=",")
