from pathlib import Path
import netCDF4
import numpy as np

from gmst_merge.useful_functions import (
    gridded_to_timeseries,
    monthly_to_annual_timeseries
)
from gmst_merge.config import DATADIR

data_file_dir = DATADIR / 'ERA5'

# softcoded filename
matches = sorted(Path(data_file_dir).glob("*.nc"))
data_file = netCDF4.Dataset(matches[-1])

mean = np.transpose(np.ma.getdata(data_file.variables['t2mn']).data);
mean = np.concatenate(
    (mean[:, 0, :], np.kron(mean[:, 1:-2, :], np.ones((1, 2, 1))), mean[:, -1, :]),
    'axis=1'
)  # convert to 1440 latitudinal bands with equal thickness
mean = monthly_to_annual_timeseries(
    gridded_to_timeseries(mean.reshape(mean.shape[0], mean.shape[1], 1)).reshape(-1, 1),
    1850
)
years = np.arange(1850, 1850 + mean.shape[0]).reshape(-1, 1)

np.savetxt(
    data_file_dir / "ensemble_time_series.csv",
    np.concatenate((years, mean), axis=1),
    fmt='%.16f',
    delimiter=","
)
