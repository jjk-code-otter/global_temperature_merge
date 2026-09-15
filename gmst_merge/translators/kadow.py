from pathlib import Path
import netCDF4
import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp, quick_plot

def convert_file():
    timestamp = get_timestamp()
    data_file_dir = DATADIR / 'Kadow'
    filename = data_file_dir / 'Kadow_et_al_2026_HadCRUT.5.1.0.0.AIinfilled.anomalies.ensemble_global_annual_mean_185001-202512.nc'
    ts_filename = data_file_dir / f'{timestamp}_Kadow_et_al_2026_HadCRUT.5.1.0.0.AIinfilled.anomalies.ensemble_global_annual_mean_185001-202512.nc'

    shutil.copy(filename, ts_filename)

    data_file = netCDF4.Dataset(filename)

    ensemble = np.transpose(data_file.variables['tas'][:, 0, 0].filled(np.nan)).reshape(-1, 1)
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
    quick_plot('Kadow', ts_out_filename, f'../Figures/BasicInputPlots/{timestamp}_Kadow.png')


if __name__ == '__main__':
    convert_file()