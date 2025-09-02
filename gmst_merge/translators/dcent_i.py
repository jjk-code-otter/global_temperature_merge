from pathlib import Path
import xarray as xa
import numpy as np
import os


def convert_file():
    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)

    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'DCENT_I'
    filename = data_file_dir / 'DCENT_DCENT-I_annual_statistics.nc'

    df = xa.open_dataset(filename)

    ntime = df.DCENT_I_GMST.shape[1]
    nensemble = df.DCENT_I_GMST.shape[0]

    output = np.zeros((ntime, nensemble + 1))
    output[:, 1:] = np.transpose(df.DCENT_I_GMST.values[:, :])
    output[:, 0] = np.arange(1850, 1850 + ntime, 1)
    output = output.astype(np.float16)

    np.savetxt(data_file_dir / "ensemble_time_series.csv", output, fmt='%.4f', delimiter=",")

