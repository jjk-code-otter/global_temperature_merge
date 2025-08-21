from pathlib import Path
import xarray as xa
import numpy as np
import os

def convert_file():

    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)


    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'GISTEMP'
    filename = data_file_dir / 'GLB.Ts+dSST.csv'

    years = []
    anoms = []

    with open(filename, 'r') as f:
        for i in range(2):
            f.readline()
        for line in f:
            if not '*' in line:
                columns = line.split(',')
                year = columns[0]
                anomaly = columns[13]

                years.append(int(year))
                anoms.append(float(anomaly))


    full_nyears = len(years)
    full_array = np.zeros((full_nyears, 201))

    full_array[:,0] = np.array(years)
    full_array[:,1:] = np.repeat(np.reshape(np.array(anoms), (full_nyears,1)), 200, axis=1)


    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'GISTEMP_ensemble'
    filename = data_file_dir / 'ensembleCombinedSeries_Global.nc'

    df = xa.open_dataset(filename)

    ntimes = df.tas.data.shape[1]
    nyears = int(ntimes / 12)

    output = np.zeros((nyears, 201))

    years = df.time.dt.year.data
    months = df.time.dt.month.data
    anomalies = df.tas.data

    output[:, 0] = np.arange(1880, 1880 + nyears, 1)
    output[:, 1:] = np.transpose(np.mean(anomalies.reshape(200, nyears, 12), axis=2))

    for i in range(200):
        full_array[0:2021-1880, i] = output[:, i] - output[2020-1880,i] + full_array[2020-1880,i]

    output = full_array

    np.savetxt(data_file_dir / "ensemble_time_series.csv", output, delimiter=",")
