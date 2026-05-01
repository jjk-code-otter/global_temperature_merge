from pathlib import Path
import os
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'COBE-STEMP3'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'COBE-STEMP3'

file = open(data_file_dir / 'annual_gm_cobe-stemp3')
while True:
    words = file.readline().split()
    if len(words) > 0:
        if words[0] == "1850":
            break
years = np.zeros((0,1))
mean = np.zeros((0,1))
uncertainty = np.zeros((0,1))
while len(words) > 0:
    years = np.append(years,float(words[0]))
    mean = np.append(mean,float(words[1]))
    uncertainty = np.append(uncertainty,float(words[2]))
    words = file.readline().split()
file.close()
years = years.reshape(-1,1)
mean = mean.reshape(-1,1)
uncertainty = uncertainty.reshape(-1,1)

np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((years,mean),axis=1), fmt='%.16f', delimiter=",")
np.savetxt(data_file_dir / "uncertainty_time_series.csv", np.concatenate((years,uncertainty),axis=1), fmt='%.16f', delimiter=",")
