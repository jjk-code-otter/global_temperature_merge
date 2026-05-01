from pathlib import Path
import os
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from useful_functions import monthly_to_annual_timeseries

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'Berkeley Earth Hires'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'Berkeley Earth Hires'

file = open(data_file_dir / 'Global_TAVG_ensemble.txt')
while True:
    words = file.readline().split()
    if len(words) > 0:
        if words[0] == "1850":
            break
ensemble = np.zeros((0,10))
while len(words) > 0:
    ensemble = np.append(ensemble,float(words[2:12]),axis=0)
    words = file.readline().split()
file.close()
ensemble = monthly_to_annual_timeseries(ensemble,1850)
year = np.arange(1850:1850+ensemble.shape[0]).reshape(-1,1)

np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((years,ensemble),axis=1), fmt='%.16f', delimiter=",")
