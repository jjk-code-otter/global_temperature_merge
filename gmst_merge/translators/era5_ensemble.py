from pathlib import Path
import netCDF4
import numpy as np
import os
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'ERA5 ensemble'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'ERA5 ensemble'

filename = data_file_dir / 'T2m_era51_time_series_from_1940.txt'

with open(filename, "rb") as f:
    num_lines = sum(1 for _ in f)

years = np.zeros(num_lines)
months = np.zeros(num_lines)
data = np.zeros((num_lines, 11))

with open(filename, 'r') as f:
    count = 0
    for line in f:
        columns = line.split()

        if len(columns) == 13:
            year = columns[0]
            month = columns[1]
            anomalies = columns[2:]
        else:
            year = columns[0][0:4]
            month = columns[0][4:]
            anomalies = columns[1:]

        # Convert the ensemble to a numpy array
        ensemble = [float(x) for x in anomalies]
        ensemble = np.array(ensemble)

        years[count] = int(year)
        months[count] = int(month)
        data[count, :] = ensemble[:]

        count += 1

nmonths = num_lines
nyears = int(nmonths / 12)

nmonths_full_years = nyears * 12

years = years[0:nmonths_full_years]
months = months[0:nmonths_full_years]
data = data[0:nmonths_full_years, :]

output = np.zeros((nyears, 12))

output[:, 0] = np.mean(years.reshape((nyears, 12)), axis=1)

for i in range(11):
    output[:, i + 1] = np.mean(data[:, i].reshape((nyears, 12)), axis=1)
    output[:, i + 1] = output[:, i + 1] - np.mean(output[:, i + 1])

np.savetxt(data_file_dir / "ensemble_time_series.csv", output, fmt='%.16f', delimiter=",")
