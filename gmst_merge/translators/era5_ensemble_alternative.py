from pathlib import Path
import netCDF4
import numpy as np
import os
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from useful_functions import gridded_to_timeseries
from useful_functions import monthly_to_annual_timeseries

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'ERA5_ensemble'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'ERA5_ensemble'

# softcoded filename
matches = sorted(Path(data_file_dir).glob("*.nc"))
data_file = netCDF4.Dataset(matches[-1])

data_file = np.transpose(np.ma.getdata(data_file.variables['t2mn']).data);
data_file = np.concatenate((ensemble[:,0,:],np.kron(mean[:,1:-2,:],np.ones((1,2,1))),ensemble[:,-1,:]),'axis=1') # convert to 1440 latitudinal bands with equal thickness

ensemble = data_file[:,:,:,0]
ensemble = monthly_to_annual_timeseries(gridded_to_timeseries(ensemble.reshape(ensemble.shape[0],ensemble.shape[1],1)).reshape(-1,1),1850)
for i in range(1,datafile.shape[3]):
	member = data_file[:,:,:,i]
	ensemble =  np.concatenate((ensemble,monthly_to_annual_timeseries(gridded_to_timeseries(member.reshape(member.shape[0],member.shape[1],1)).reshape(-1,1),1850)),'axis=1')
years = np.arange(1850,1850+ensemble.shape[0]).reshape(-1,1)

np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((years,ensemble),axis=1), fmt='%.16f', delimiter=",")