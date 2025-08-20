#  Global Temperature Merge - a package for merging global temperature datasets.
#  Copyright \(c\) 2025 John Kennedy
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path
import numpy as np
import os

def convert_file():

    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)

    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'ERA5'
    filename = data_file_dir / 'C3S_Bulletin_temp_202507_Fig1b_timeseries_anomalies_ref1991-2020_global_allmonths_data.csv'

    years = []
    anoms = []
    with open(filename, 'r') as f:
        for i in range(12):
            f.readline()
        for line in f:
            columns = line.split(',')
            time = columns[0].split('-')
            years.append(int(time[0]))
            anoms.append(float(columns[3]))

    years = np.array(years)
    anoms = np.array(anoms)

    nyears = int(len(anoms)/12)

    years = np.mean(years.reshape(nyears, 12), axis=1)
    anoms = np.mean(anoms.reshape(nyears, 12), axis=1)

    output = np.zeros((nyears, 2))

    output[:,0] = years[:]
    output[:,1] = anoms[:]

    np.savetxt(data_file_dir / "ensemble_time_series.csv", output, delimiter=",")

    output = np.zeros((nyears, 2))

    output[:,0] = years[:]
    output[:,1] = anoms[:] * 0.0 + 0.03

    np.savetxt(data_file_dir / "uncertainty_time_series.csv", output, delimiter=",")