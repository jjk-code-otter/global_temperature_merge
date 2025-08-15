from pathlib import Path
import os

def convert_file():

    data_dir_env = os.getenv('DATADIR')
    DATA_DIR = Path(data_dir_env)

    data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'Berkeley Earth'
    filename = data_file_dir / 'Land_and_Ocean_summary.txt'

    with open(data_file_dir / 'ensemble_time_series.csv', 'w') as o:
        with open(filename, 'r') as f:
            for i in range(58):
                f.readline()
            for line in f:
                columns = line.split()
                columns = columns[0:2]
                line = ','.join(columns) + '\n'
                o.write(line)

    with open(data_file_dir / 'uncertainty_time_series.csv', 'w') as o:
        with open(filename, 'r') as f:
            for i in range(58):
                f.readline()
            for line in f:
                columns = line.split()
                columns = [columns[0], f'{float(columns[2])/1.96:.4f}']  # 95% confidence intervals
                line = ','.join(columns) + '\n'
                o.write(line)
