import netCDF4
import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp

def convert_file():
    timestamp = get_timestamp()
    data_file_dir = DATADIR / 'GloSAT'

    # https://www.metoffice.gov.uk/hadobs/glosatref/data/GloSATref.1.0.0.0/analysis/diagnostics/GloSATref.1.0.0.0.analysis.ensemble_series.global.annual.nc
    filename = data_file_dir / 'GloSATref.1.0.0.0.analysis.ensemble_series.global.annual.nc'
    ts_filename = data_file_dir / f'{timestamp}_GloSATref.1.0.0.0.analysis.ensemble_series.global.annual.nc'

    shutil.copy(filename, ts_filename)

    # softcoded filename
    data_file = netCDF4.Dataset(filename)

    ensemble = np.transpose(data_file.variables['tas'][:].filled(np.nan))
    coverage_unc = np.transpose(data_file.variables['coverage_unc'][:].filled(np.nan))
    scaling_factor = np.divide(np.sqrt(np.square(coverage_unc) + np.var(ensemble, axis=1, ddof=1)),
                               np.std(ensemble, axis=1, ddof=1)).reshape(-1, 1)
    mean = np.mean(ensemble, axis=1).reshape(-1, 1)
    ensemble = np.multiply(ensemble - mean, scaling_factor) + mean
    years = np.arange(1781, 1781 + ensemble.shape[0]).reshape(-1, 1)

    years = years[1850-1781:,:]
    ensemble = ensemble[1850-1781:,:]

    out_filename = data_file_dir / "ensemble_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_ensemble_time_series.csv"
    np.savetxt(
        out_filename,
        np.concatenate((years, ensemble), axis=1),
        fmt='%.16f',
        delimiter=","
    )
    shutil.copy(out_filename, ts_out_filename)

if __name__ == '__main__':
    convert_file()