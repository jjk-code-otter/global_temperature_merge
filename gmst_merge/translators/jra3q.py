import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp, quick_plot


def convert_file():
    timestamp = get_timestamp()
    data_file_dir = DATADIR / 'JRA-3Q'

    filename = data_file_dir / 'JRA-3Q_tmp2m_global_ts_125_Clim9120.txt'
    ts_filename = data_file_dir / f'{timestamp}_JRA-3Q_tmp2m_global_ts_125_Clim9120.txt'
    shutil.copy(filename, ts_filename)

    years = []
    anoms = []
    with open(filename, 'r') as f:
        for i in range(3):
            f.readline()
        for line in f:
            columns = line.split()
            time = columns[0].split('-')
            years.append(int(time[0]))
            anoms.append(float(columns[1]))

    years = np.array(years)
    anoms = np.array(anoms)

    # Count out full years only
    nyears = int(len(anoms) / 12)

    years = years[:nyears * 12]
    anoms = anoms[:nyears * 12]

    years = np.mean(years.reshape(nyears, 12), axis=1)
    anoms = np.mean(anoms.reshape(nyears, 12), axis=1)

    output = np.zeros((nyears, 2))

    output[:, 0] = years[:]
    output[:, 1] = anoms[:]

    out_filename = data_file_dir / "ensemble_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_ensemble_time_series.csv"
    np.savetxt(
        out_filename,
        output,
        delimiter=","
    )
    shutil.copy(out_filename, ts_out_filename)
    quick_plot('JRA3Q', ts_out_filename, f'../Figures/BasicInputPlots/{timestamp}_JRA3Q.png')


if __name__ == '__main__':
    convert_file()
