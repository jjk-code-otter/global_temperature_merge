import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp


def convert_file():
    timestamp = get_timestamp()

    # received by email
    data_file_dir = DATADIR / 'Cooper'
    filename = data_file_dir / 'cooper_etal_2025_annglobmean_200ens.csv'
    ts_filename = data_file_dir / f'{timestamp}_cooper_etal_2025_annglobmean_200ens.csv'

    shutil.copy(filename, ts_filename)

    nyears = 2023 - 1850 + 1
    nensemble = 200

    with open(filename, 'r') as f:
        f.readline()

        data = np.zeros((nyears, nensemble))
        count = 0

        for line in f:
            columns = line.split(",")
            columns = columns[2:]
            years_of_data = np.array([float(x) for x in columns])
            data[:, count] = years_of_data[:]
            count += 1

    time = np.arange(1850, 1850 + nyears, 1)

    output = np.zeros((nyears, nensemble + 1))

    output[:, 0] = time[:]
    output[:, 1:] = data[:]

    out_filename = data_file_dir / f"ensemble_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_ensemble_time_series.csv"
    np.savetxt(
        out_filename,
        output,
        fmt='%.4f',
        delimiter=","
    )

    shutil.copy(out_filename, ts_out_filename)


if __name__ == '__main__':
    convert_file()