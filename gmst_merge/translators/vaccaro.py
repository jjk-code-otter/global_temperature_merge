from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

def make_xarray(target_grid, times, latitudes, longitudes, variable: str = 'tas_mean') -> xa.Dataset:
    ds = xa.Dataset({
        variable: xa.DataArray(
            data=target_grid,
            dims=['time', 'latitude', 'longitude'],
            coords={'time': times, 'latitude': latitudes, 'longitude': longitudes},
            attrs={'long_name': '2m air temperature', 'units': 'K'}
        )
    },
        attrs={'project': 'NA'}
    )

    return ds

def convert_file_long():

    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)

    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'Vaccaro'

    n_ensemble = 100

    ntime=2016
    nyears = int(ntime/12)
    output = np.zeros((nyears,n_ensemble+1))

    for i in range(n_ensemble):
        filename = data_file_dir / f'HadCRUT4.6.0.0.anomalies.{i + 1}_GraphEM_SP60.nc'
        print(filename)

        df = xa.open_dataset(filename)

        number_of_months = df.temperature_anomaly.data.shape[0]
        latitudes = np.linspace(-87.5, 87.5, 36)
        longitudes = np.linspace(-177.5, 177.5, 72)
        times = pd.date_range(start=f'1850-01-01', freq='1MS', periods=number_of_months)

        target_grid = np.zeros((number_of_months, 36, 72))
        target_grid[:, :, :] = df.temperature_anomaly.data[:, :, :]

        df = make_xarray(target_grid, times, latitudes, longitudes)

        # Open file get area weights
        weights = np.cos(np.deg2rad(df.tas_mean.latitude))

        # Calculate the area-weighted average, then the annual average
        regional_ts = df.tas_mean.weighted(weights).mean(dim=("latitude", "longitude"))
        regional_ts = regional_ts.data
        regional_ts = np.mean(regional_ts.reshape(nyears, 12), axis=1)

        # Make a time axis
        time = np.arange(1850,1850+nyears, 1)

        output[:,0] = time[:]
        output[:,i+1] = regional_ts[:]

    np.savetxt(data_file_dir / "ensemble_time_series.csv", output, delimiter=",")

