import xarray as xa
import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp, quick_plot


def convert_file():
    timestamp = get_timestamp()
    data_file_dir = DATADIR / 'DCENT_I'
    filename = data_file_dir / 'DCENT_DCENT-I_annual_statistics.nc'
    ts_filename = data_file_dir / f'{timestamp}_DCENT_DCENT-I_annual_statistics.nc'
    shutil.copy(filename, ts_filename)

    df = xa.open_dataset(filename)

    ntime = df.DCENT_I_GMST.shape[1]
    nensemble = df.DCENT_I_GMST.shape[0]

    output = np.zeros((ntime, nensemble + 1))
    output[:, 1:] = np.transpose(df.DCENT_I_GMST.values[:, :])
    output[:, 0] = np.arange(1850, 1850 + ntime, 1)
    output = output.astype(np.float16)

    out_filename = data_file_dir / "ensemble_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_ensemble_time_series.csv"
    np.savetxt(
        out_filename,
        output,
        fmt='%.4f',
        delimiter=","
    )
    shutil.copy(out_filename, ts_out_filename)
    quick_plot('DCENT', ts_out_filename, f'../Figures/BasicInputPlots/{timestamp}_DCENT.png')


if __name__ == '__main__':
    convert_file()
