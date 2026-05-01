from pathlib import Path
import netCDF4
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'HadCRU_MLE'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'HadCRU_MLE'

# softcoded filename
matches = sorted(Path(data_file_dir).glob("HadCRU_MLE_v*_timeseries_annual_anomalies_ensemble.nc"))
data_file = netCDF4.Dataset(matches[-1])

ensemble = np.transpose(np.ma.getdata(data_file.variables['surface_temperature_anomaly']).data)
years = np.arange(1850,1850+ensemble.shape[0]).reshape((-1,1))

np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((years,ensemble),axis=1), fmt='%.16f', delimiter=",")

