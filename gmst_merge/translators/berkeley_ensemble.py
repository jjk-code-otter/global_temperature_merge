import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp, quick_plot


def convert_file():
    timestamp = get_timestamp()

    # https://storage.googleapis.com/berkeley-earth-temperature-hr/global/Global_TAVG_ensemble.txt
    data_file_dir = DATADIR / 'Berkeley Earth Hires'
    filename = data_file_dir / 'Global_TAVG_ensemble.txt'
    ts_filename = data_file_dir / f'{timestamp}_Global_TAVG_ensemble.txt'

    shutil.copy(filename, ts_filename)

    nyears = 2025 - 1850 + 1
    nmonths = 12 * nyears
    nensemble = 10

    with open(filename, 'r') as f:
        for i in range(49):
            f.readline()

        years = []
        months = []

        data = np.zeros((nmonths, nensemble))
        count = 0

        for line in f:
            columns = line.split()
            year = int(columns[0])
            years.append(year)
            months.append(int(columns[1]))

            columns = columns[2:]
            ensemble_members = np.array([float(x) for x in columns])

            if year < 2026:
                data[count, :] = ensemble_members[:]
                count += 1

    data = np.mean(data.reshape(nyears, 12, 10), axis=1)
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
    quick_plot('Berkeley Earth Hires', ts_out_filename, f'../Figures/BasicInputPlots/{timestamp}_Berkeley_Earth_Hires.png')


if __name__ == '__main__':
    convert_file()