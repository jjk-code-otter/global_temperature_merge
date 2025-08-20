from pathlib import Path
import xarray as xa
import numpy as np
import os

def convert_file():

    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)

    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'GETQUOCS'
    filename = data_file_dir / 'GETQUOCS_ensemble_subsample_monthly.packed.nc'

    # Open file get area weights
    df = xa.open_dataset(filename)
    weights = np.cos(np.deg2rad(df.temperature.latitude))

    # Calculate the area-weighted average, then the annual average
    regional_ts = df.temperature.weighted(weights).mean(dim=("latitude", "longitude"))
    regional_ts = regional_ts.data
    regional_ts = np.mean(regional_ts.reshape(100, int(2028/12), 12), axis=2)

    # Make a time axis
    time = np.arange(1850,1850+169, 1)

    # Transpose array and add time axis for writing
    regional_ts = np.transpose(regional_ts)
    output = np.zeros((169,101))
    output[:,0] = time[:]
    output[:,1:] = regional_ts[:,:]

    np.savetxt(data_file_dir / "ensemble_time_series.csv", output, delimiter=",")

