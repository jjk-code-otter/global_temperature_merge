from pathlib import Path
import os

def convert_file():

    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)

    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'NOAA v6'
    filename = data_file_dir / 'aravg.ann.land_ocean.90S.90N.v6.0.0.202412.asc'

    with open(data_file_dir / 'ensemble_time_series.csv', 'w') as o:
        with open(filename, 'r') as f:
            for line in f:
                columns = line.split()
                columns = columns[0:2]
                line = ','.join(columns) + '\n'
                o.write(line)
