from pathlib import Path
import numpy as np
import os

def convert_file():
    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)

    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'COBE-STEMP3'
    filename = data_file_dir / 'gm_cobe-stemp3'

    years = []
    anoms = []
    uncertainties = []
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            columns = line.split()
            years.append(int(columns[0]))
            anoms.append(float(columns[2]))
            uncertainties.append(float(columns[3]))

    years = np.array(years)
    anoms = np.array(anoms)
    uncertainties = np.array(uncertainties)

    nyears = int(len(anoms)/12)

    years = np.mean(years.reshape(nyears, 12), axis=1)
    anoms = np.mean(anoms.reshape(nyears, 12), axis=1)
    uncertainties = np.mean(uncertainties.reshape(nyears, 12), axis=1)

    output = np.zeros((nyears, 2))

    output[:,0] = years[:]
    output[:,1] = anoms[:]

    np.savetxt(data_file_dir / "ensemble_time_series.csv", output, fmt='%.4f', delimiter=",")

    output = np.zeros((nyears, 2))

    output[:,0] = years[:]
    output[:,1] = uncertainties[:]

    np.savetxt(data_file_dir / "uncertainty_time_series.csv", output, fmt='%.4f', delimiter=",")