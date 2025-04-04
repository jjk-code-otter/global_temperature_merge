from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

# http://www.gwpu.net/h-nd-155.html
hadcrut5_dir = DATA_DIR / 'ManagedData' / 'Data' / 'CMST'
filename = hadcrut5_dir / 'annual_data.csv'

df = pd.read_csv(filename)

anomalies = df.Global
times = df.year
uncertainties = df.GMST

ntimes = len(anomalies)
nyears = len(times)
nuncs = len(uncertainties)

output = np.zeros((nyears, 2))

output[:, 0] = times
output[:, 1] = anomalies

np.savetxt(hadcrut5_dir / "ensemble_time_series.csv", output, fmt='%.4f', delimiter=",")

output = np.zeros((nyears, 2))

output[:, 0] = times
output[:, 1] = uncertainties

np.savetxt(hadcrut5_dir / "uncertainty_time_series.csv", output, fmt='%.4f', delimiter=",")
