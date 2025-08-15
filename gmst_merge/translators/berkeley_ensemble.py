from pathlib import Path
import numpy as np
import os


def convert_file():

    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)

    # https://storage.googleapis.com/berkeley-earth-temperature-hr/global/Global_TAVG_ensemble.txt
    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'Berkeley Earth Hires'
    filename = data_file_dir / 'Global_TAVG_ensemble.txt'

    nyears = 2024-1850+1
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

            if year < 2025:
                data[count, :] = ensemble_members[:]
                count += 1

    data = np.mean(data.reshape(nyears, 12, 10), axis=1)
    time = np.arange(1850, 1850+nyears, 1)

    output = np.zeros((nyears,nensemble+1))

    output[:,0] = time[:]
    output[:,1:] = data[:]

    np.savetxt(data_file_dir / "ensemble_time_series.csv", output, fmt='%.4f', delimiter=",")

