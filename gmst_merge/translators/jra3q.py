from pathlib import Path
import numpy as np
import os


def convert_file():
    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)

    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'JRA-3Q'
    filename = data_file_dir / 'JRA-3Q_tmp2m_global_ts_125_Clim9120.txt'

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

    nyears = int(len(anoms) / 12)

    years = years[:nyears * 12]
    anoms = anoms[:nyears * 12]

    years = np.mean(years.reshape(nyears, 12), axis=1)
    anoms = np.mean(anoms.reshape(nyears, 12), axis=1)

    output = np.zeros((nyears, 2))

    output[:, 0] = years[:]
    output[:, 1] = anoms[:]

    np.savetxt(data_file_dir / "ensemble_time_series.csv", output, delimiter=",")

    output = np.zeros((nyears, 2))

    output[:, 0] = years[:]
    output[:, 1] = anoms[:] * 0.0 + 0.06

    np.savetxt(data_file_dir / "uncertainty_time_series.csv", output, delimiter=",")
