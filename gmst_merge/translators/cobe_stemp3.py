from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

hadcrut5_dir = DATA_DIR / 'ManagedData' / 'Data' / 'COBE-STEMP3'
filename = hadcrut5_dir / 'gm_cobe-stemp3'

years = []
anoms = []
with open(filename, 'r') as f:
    f.readline()
    for line in f:
        columns = line.split()
        years.append(int(columns[0]))
        anoms.append(float(columns[2]))

years = np.array(years)
anoms = np.array(anoms)

nyears = int(len(anoms)/12)

years = np.mean(years.reshape(nyears, 12), axis=1)
anoms = np.mean(anoms.reshape(nyears, 12), axis=1)

output = np.zeros((nyears, 2))

output[:,0] = years[:]
output[:,1] = anoms[:]

np.savetxt(hadcrut5_dir / "ensemble_time_series.csv", output, delimiter=",")