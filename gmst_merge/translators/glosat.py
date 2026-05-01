from pathlib import Path
import netCDF4
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'GloSAT'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'GloSAT'

# softcoded filename
matches = sorted(Path(data_file_dir).glob("GloSATref-*_analysis_ensemble-series_global_annual.nc"))
data_file = netCDF4.Dataset(matches[-1])

ensemble = np.transpose(data_file.variables['tas'][:].filled(np.nan))
coverage_unc = np.transpose(data_file.variables['coverage_unc'][:].filled(np.nan))
scaling_factor = np.divide(np.sqrt(np.square(coverage_unc)+np.var(ensemble,axis=1,ddof=1)),np.std(ensemble,axis=1,ddof=1)).reshape(-1,1)
mean = np.mean(ensemble,axis=1).reshape(-1,1);
ensemble = np.multiply(ensemble-mean,scaling_factor)+mean;
years = np.arange(1781,1781+ensemble.shape[0]).reshape(-1,1)

np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((years,ensemble),axis=1), fmt='%.16f', delimiter=",")
