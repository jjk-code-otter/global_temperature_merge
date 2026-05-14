import netCDF4
import numpy as np

from gmst_merge.config import DATADIR

data_file_dir = DATADIR / 'DCENT-I'

data_file = netCDF4.Dataset('DCENT_DCENT-I_annual_statistics.nc')

ensemble = np.transpose(np.ma.getdata(data_file.variables['DCENT_I_GMST']).data)
years = np.arange(1850, 1850 + ensemble.shape[0]).reshape((-1, 1))

np.savetxt(
    data_file_dir / "ensemble_time_series.csv",
    np.concatenate((years, ensemble), axis=1),
    fmt='%.16f',
    delimiter=","
)
