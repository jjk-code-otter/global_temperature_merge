import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp


def convert_file():
    timestamp = get_timestamp()

    data_file_dir = DATADIR / 'Berkeley Earth'
    filename = data_file_dir / 'Land_and_Ocean_summary.txt'
    ts_filename = data_file_dir / f'{timestamp}_Land_and_Ocean_summary.txt'

    shutil.copy(filename, ts_filename)

    out_filename = data_file_dir / f'ensemble_time_series.csv'
    ts_out_filename = data_file_dir / f'{timestamp}_ensemble_time_series.csv'
    with open(out_filename, 'w') as o:
        with open(ts_filename, 'r') as f:
            for i in range(58):
                f.readline()
            for line in f:
                columns = line.split()
                columns = columns[0:2]
                line = ','.join(columns) + '\n'
                o.write(line)

    shutil.copy(out_filename, ts_out_filename)

    out_filename = data_file_dir / f'uncertainty_time_series.csv'
    ts_out_filename = data_file_dir / f'{timestamp}_uncertainty_time_series.csv'
    with open(out_filename, 'w') as o:
        with open(ts_filename, 'r') as f:
            for i in range(58):
                f.readline()
            for line in f:
                columns = line.split()
                columns = [columns[0], f'{float(columns[2]) / 1.96:.4f}']  # 95% confidence intervals
                line = ','.join(columns) + '\n'
                o.write(line)

    shutil.copy(out_filename, ts_out_filename)

if __name__ == '__main__':
    convert_file()
