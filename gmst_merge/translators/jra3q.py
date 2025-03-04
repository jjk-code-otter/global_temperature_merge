from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

hadcrut5_dir = DATA_DIR / 'ManagedData' / 'Data' / 'JRA-3Q'
filename = hadcrut5_dir / 'JRA-3Q_tmp2m_global_ts_125_Clim9120.txt'

years = []
anoms = []
with open(filename, 'r') as f:
    for i in range(3):
        f.readline()
    for line in f:
        columns = line.split()
        time = columns[0].split('-')
        years.append(int(time[0]))
        anoms.append(float(columns[1]))

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