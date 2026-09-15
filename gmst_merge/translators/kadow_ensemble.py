from pathlib import Path
import xarray as xa
import numpy as np
import shutil
import os

from gmst_merge.config import DATADIR, get_timestamp, quick_plot


def convert_file_long():
    timestamp = get_timestamp()
    data_file_dir = DATADIR / 'Kadow_ensemble'

    n_ensemble = 200
    n_months = 2074
    n_years = int(n_months / 12)
    n_months_whole = n_years * 12

    output = np.zeros((n_years, n_ensemble + 1))

    for member in range(n_ensemble):
        filename = data_file_dir / f'20crtaspadzens_tas_mon-gl-72x36_hadcrut5_observation_ens-3_1850-2022_image_{member + 1}.nc'

        print(filename)

        # Open file get area weights
        df = xa.open_dataset(filename)
        weights = np.cos(np.deg2rad(df.tas.latitude))

        # Calculate the area-weighted average, then the annual average
        regional_ts = df.tas.weighted(weights).mean(dim=("latitude", "longitude"))
        regional_ts = regional_ts.data
        regional_ts = regional_ts[0:n_months_whole].astype(np.float16)
        regional_ts = np.mean(regional_ts.reshape(n_years, 12), axis=1)

        # Make a time axis
        time = np.arange(1850, 1850 + n_years, 1)

        output[:, 0] = time[:]
        output[:, member + 1] = regional_ts[:]

    out_filename = data_file_dir / f"ensemble_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_ensemble_time_series.csv"
    np.savetxt(
        out_filename,
        output,
        fmt='%.16f',
        delimiter=","
    )
    shutil.copy(out_filename, ts_out_filename)
    quick_plot('Kadow ensemble', ts_out_filename, f'../Figures/BasicInputPlots/{timestamp}_Kadow_ensemble.png')


if __name__ == '__main__':
    convert_file_long()
