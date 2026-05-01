from pathlib import Path
import netCDF4
import numpy as np
import os
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from useful_functions import monthly_to_annual_timeseries

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'ERA5'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'ERA5'

# softcoded filename
matches = sorted(Path(data_file_dir).glob("C3S_Bulletin_temp_*_timeseries_anomalies_ref1850-1900_global_allmonths_data.csv"))
data_file = netCDF4.Dataset(matches[-1])

mean = pd.read_csv(data_file,usecols=[0])
first_row = np.where(mean=="month")[0][0]+1
mean = pd.read_csv(data_file,skiprows=first_row+1).to_numpy()
mean = mean[:,1]
mean = monthly_to_annual_timeseries(mean,1940)
years = np.arange(1940,1940+mean.shape[0]).reshape((-1,1))

np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((years,mean),axis=1), fmt='%.16f', delimiter=",")
