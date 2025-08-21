from pathlib import Path
import os


def convert_file():
    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)

    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'GISTEMP'
    filename = data_file_dir / 'GLB.Ts+dSST.csv'

    with open(data_file_dir / 'ensemble_time_series.csv', 'w') as o:
        with open(filename, 'r') as f:
            for i in range(2):
                f.readline()
            for line in f:
                if "*" not in line:
                    columns = line.split(',')
                    year = columns[0]
                    anomaly = columns[13]
                    columns = [year, anomaly]
                    line = ','.join(columns) + '\n'
                    o.write(line)
