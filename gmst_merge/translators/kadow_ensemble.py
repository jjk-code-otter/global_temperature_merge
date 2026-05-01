from pathlib import Path
import netCDF4
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from useful_functions import gridded_to_timeseries
from useful_functions import monthly_to_annual_timeseries

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'Kadow_ensemble'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'Kadow_ensemble'


ensemble = [];
for i in range(5):
    for j in range(200):
        # softcoded filename
        matches = sorted(Path(data_file_dir).glob("*.nc"))
        data_file = netCDF4.Dataset(matches[-1])

        # softcoded filename
        matches = sorted(Path(data_file_dir).glob(f'20crtaspadzens_tas_mon-gl-72x36_hadcrut5_observation_ens-{i+1}_1850-*_image_{j+1}.nc'))
        data_file = netCDF4.Dataset(matches[-1])
        data_file = np.transpose(np.ma.getdata(data_file.variables['tas']).data);
        if ensemble == []:
            ensemble = monthly_to_annual_timeseries(gridded_to_timeseries(data_file.reshape(member.shape[0],member.shape[1],1)).reshape(-1,1),1850);
        else:
            ensemble =  np.concatenate((ensemble,monthly_to_annual_timeseries(gridded_to_timeseries(member.reshape(member.shape[0],member.shape[1],1)).reshape(-1,1),1850)),'axis=1')
years = np.arange(1850,1850+ensemble.shape[0]).reshape(-1,1)

np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((years,ensemble),axis=1), fmt='%.16f', delimiter=",")