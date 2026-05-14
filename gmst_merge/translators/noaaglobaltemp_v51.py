from pathlib import Path
import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp

data_file_dir = DATADIR / 'NOAA v5.1'


def convert_file():
    timestamp = get_timestamp()

    filename = data_file_dir / "aravg.ann.land_ocean.90S.90N.v5.1.0.202312.asc"
    ts_filename = data_file_dir / f"{timestamp}_aravg.ann.land_ocean.90S.90N.v5.1.0.202312.asc"

    shutil.copy(filename, ts_filename)

    with open(filename) as file:
        years = np.zeros((0, 1))
        mean = np.zeros((0, 1))
        while True:
            words = file.readline().split()
            if len(words) == 0:
                break
            years = np.append(years, float(words[0]))
            mean = np.append(mean, float(words[1]))

    years = years.reshape(-1, 1)
    mean = mean.reshape(-1, 1)

    out_filename = data_file_dir / "ensemble_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_ensemble_time_series.csv"
    np.savetxt(
        out_filename,
        np.concatenate((years, mean), axis=1),
        fmt='%.16f',
        delimiter=","
    )
    shutil.copy(out_filename, ts_out_filename)


if __name__ == '__main__':
    convert_file()
