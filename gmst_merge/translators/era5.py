from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

hadcrut5_dir = DATA_DIR / 'ManagedData' / 'Data' / 'ERA5'
filename = hadcrut5_dir / 'C3S_Bulletin_temp_202411_Fig1b_timeseries_anomalies_ref1991-2020_global_allmonths_data.csv'

years = []
anoms = []
with open(filename, 'r') as f:
    for i in range(12):
        f.readline()
    for line in f:
        columns = line.split(',')
        time = columns[0].split('-')
        years.append(int(time[0]))
        anoms.append(float(columns[3]))

years = np.array(years)
anoms = np.array(anoms)

nyears = int(len(anoms)/12)

years = np.mean(years.reshape(nyears, 12), axis=1)
anoms = np.mean(anoms.reshape(nyears, 12), axis=1)

output = np.zeros((nyears, 2))

output[:,0] = years[:]
output[:,1] = anoms[:]

np.savetxt(hadcrut5_dir / "ensemble_time_series.csv", output, delimiter=",")

output = np.zeros((nyears, 2))

output[:,0] = years[:]
output[:,1] = anoms[:] * 0.0 + 0.03

np.savetxt(hadcrut5_dir / "uncertainty_time_series.csv", output, delimiter=",")