from pathlib import Path
import xarray as xa
import numpy as np
import os

def convert_file():

    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)

    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'Kadow2025'
    filename = data_file_dir / 'Kadow_et_al_2026_HadCRUT.5.1.0.0.AIinfilled.anomalies.ensemble_global_annual_mean_185001-202512.nc'

    df = xa.open_dataset(filename)

    ntimes = df.tas.data.shape[0]

    output = np.zeros((ntimes, 2))

    years = df.time.dt.year.data
    anomalies = np.reshape(df.tas.data, (ntimes))

    output[:,0] = np.arange(1850,2026,1)
    output[:,1] = anomalies[:]

    np.savetxt(data_file_dir / "ensemble_time_series.csv", output, delimiter=",")

if __name__ == '__main__':
    convert_file()