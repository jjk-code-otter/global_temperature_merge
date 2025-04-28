from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'GISTEMP'
filename = data_file_dir / 'GLB.Ts+dSST.csv'

with open(data_file_dir / 'ensemble_time_series.csv', 'w') as o:
    with open(filename, 'r') as f:
        for i in range(2):
            f.readline()
        for line in f:
            columns = line.split(',')
            year = columns[0]
            anomaly = columns[13]
            columns = [year, anomaly]
            line = ','.join(columns) + '\n'
            o.write(line)
