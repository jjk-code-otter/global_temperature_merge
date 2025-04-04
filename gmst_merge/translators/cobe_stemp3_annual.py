from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

hadcrut5_dir = DATA_DIR / 'ManagedData' / 'Data' / 'COBE-STEMP3'
filename = hadcrut5_dir / 'annual_gm_cobe-stemp3.bin'

years = []
anoms = []
uncertainties = []
with open(filename, 'r') as f:
    f.readline()
    for line in f:
        columns = line.split()
        years.append(int(columns[0]))
        anoms.append(float(columns[1]))
        uncertainties.append(float(columns[2]))

years = np.array(years)
anoms = np.array(anoms)
uncertainties = np.array(uncertainties)

nyears = len(anoms)

output = np.zeros((nyears, 2))

output[:,0] = years[:]
output[:,1] = anoms[:]

np.savetxt(hadcrut5_dir / "ensemble_time_series.csv", output, fmt='%.4f', delimiter=",")

output = np.zeros((nyears, 2))

output[:,0] = years[:]
output[:,1] = uncertainties[:]

np.savetxt(hadcrut5_dir / "uncertainty_time_series.csv", output, fmt='%.4f', delimiter=",")