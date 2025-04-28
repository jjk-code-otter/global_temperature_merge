from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

data_file_dir = DATA_DIR / 'ManagedData' / 'Data' / 'GloSAT'
filename = data_file_dir / 'GloSATref.1.0.0.0.analysis.ensemble_series.global.annual.csv'

with open(data_file_dir / 'ensemble_time_series.csv', 'w') as o:
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            columns = line.split(',')
            year = [columns[0]]
            ensemble = columns[3:]
            columns = year + ensemble
            line = ','.join(columns)
            if int(year[0]) >= 1850:
                o.write(line)
