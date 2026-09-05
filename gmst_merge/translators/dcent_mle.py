from pathlib import Path
import netCDF4
import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp

def convert_file():
    timestamp = get_timestamp()
    data_file_dir = DATADIR / 'DCENT_MLE'
    filename = data_file_dir / 'DCENT_MLE_v1.2_timeseries_annual_anomalies_ensemble.nc'
    ts_filename = data_file_dir / f'{timestamp}_DCENT_MLE_v1.2_timeseries_annual_anomalies_ensemble.nc'

    shutil.copy(filename, ts_filename)

    data_file = netCDF4.Dataset(filename)

    ensemble = np.transpose(np.ma.getdata(data_file.variables['surface_temperature_anomaly']).data)
    years = np.arange(1850, 1850 + ensemble.shape[0]).reshape((-1, 1))

    out_filename = data_file_dir / "ensemble_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_ensemble_time_series.csv"
    np.savetxt(out_filename, np.concatenate((years, ensemble), axis=1), fmt='%.16f',
               delimiter=",")
    shutil.copy(out_filename, ts_out_filename)

if __name__ == '__main__':
    convert_file()