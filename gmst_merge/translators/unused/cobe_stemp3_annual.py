import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp


def convert_file():
    timestamp = get_timestamp()
    # https://climate.mri-jma.go.jp/pub/archives/Ishii-et-al_COBE-SST3/gm/annual_gm_cobe-stemp3
    # columns are: year, STEMP, 1-sigma err.
    data_file_dir = DATADIR / 'COBE-STEMP3'
    filename = data_file_dir / 'annual_gm_cobe-stemp3.txt'
    ts_filename = data_file_dir / f'{timestamp}_annual_gm_cobe-stemp3.txt'

    shutil.copyfile(filename, ts_filename)

    years = []
    anoms = []
    uncertainties = []
    with open(filename, 'r') as f:
        f.readline()
        f.readline()
        for line in f:
            columns = line.split()
            years.append(int(columns[0]))
            anoms.append(float(columns[1]))
            uncertainties.append(float(columns[2]))

    years = np.array(years)
    anoms = np.array(anoms)
    uncertainties = np.array(uncertainties)

    nyears = len(anoms)

    output = np.zeros((nyears, 2))

    output[:, 0] = years[:]
    output[:, 1] = anoms[:]

    out_filename = data_file_dir / "ensemble_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_ensemble_time_series.csv"
    np.savetxt(
        out_filename,
        output,
        fmt='%.4f',
        delimiter=","
    )
    shutil.copy(out_filename, ts_out_filename)

    output = np.zeros((nyears, 2))

    output[:, 0] = years[:]
    output[:, 1] = uncertainties[:]

    out_filename = data_file_dir / "uncertainty_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_uncertainty_time_series.csv"
    np.savetxt(
        out_filename,
        output,
        fmt='%.4f',
        delimiter=","
    )
    shutil.copy(out_filename, ts_out_filename)

if __name__ == '__main__':
    convert_file()
