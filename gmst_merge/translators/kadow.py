from pathlib import Path
import netCDF4
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'Kadow'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'Kadow'

# softcoded filename
matches = sorted(Path(data_file_dir).glob("Kadow_et_al_*_HadCRUT.*.AIinfilled.anomalies.ensemble_global_annual_mean_*.nc"))
data_file = netCDF4.Dataset(matches[-1])

mean = np.transpose(np.ma.getdata(data_file.variables['tas_mean']).data)
years = np.arange(1850,1850+mean.shape[0]).reshape((-1,1))

np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((years,mean),axis=1), fmt='%.16f', delimiter=",")