from pathlib import Path
import netCDF4
import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp

def convert_file():
    timestamp = get_timestamp()
    data_file_dir = DATADIR / 'HadCRUT5'
    filename = data_file_dir / f'HadCRUT.5.1.0.0.analysis.ensemble_series.global.annual.nc'
    ts_filename = data_file_dir / f'{timestamp}_HadCRUT.5.1.0.0.analysis.ensemble_series.global.annual.nc'

    shutil.copy(filename, ts_filename)

    data_file = netCDF4.Dataset(filename)

    ensemble = np.transpose(data_file.variables['tas'][:].filled(np.nan))
    coverage_unc = np.transpose(data_file.variables['coverage_unc'][:].filled(np.nan))
    scaling_factor = np.divide(np.sqrt(np.square(coverage_unc)+np.var(ensemble,axis=1,ddof=1)),np.std(ensemble,axis=1,ddof=1)).reshape(-1,1)
    mean = np.mean(ensemble,axis=1).reshape(-1,1)
    ensemble = np.multiply(ensemble-mean,scaling_factor)+mean
    years = np.arange(1850,1850+ensemble.shape[0]).reshape(-1,1)

    out_filename = data_file_dir / "ensemble_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_ensemble_time_series.csv"
    np.savetxt(
        out_filename,
        np.concatenate((years,ensemble),axis=1),
        fmt='%.16f',
        delimiter=","
    )
    shutil.copy(out_filename, ts_out_filename)

if __name__ == '__main__':
    convert_file()