from pathlib import Path
import numpy as np
import os
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'CMST3'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'CMST3'

data_file = pd.read_excel(data_file_dir / 'China-MST3.0-Imax.xlsx').to_numpy()
first_row = np.where(data_file[:,0]==1850)[0][0]
years = data_file[first_row:,0].reshape(-1,1)
mean = data_file[first_row:,1].reshape(-1,1)
uncertainty = data_file[first_row:,7].reshape(-1,1)

np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((years,mean),axis=1), fmt='%.16f', delimiter=",")
np.savetxt(data_file_dir / "uncertainty_time_series.csv", np.concatenate((years,uncertainty),axis=1), fmt='%.16f', delimiter=",")