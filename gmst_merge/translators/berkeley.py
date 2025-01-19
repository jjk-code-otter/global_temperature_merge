from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

berkeley_name = DATA_DIR / 'ManagedData' / 'Data' / 'Berkeley Earth'
filename = berkeley_name / 'Land_and_Ocean_summary.txt'

with open(berkeley_name / 'ensemble_time_series.csv', 'w') as o:
    with open(filename, 'r') as f:
        for i in range(58):
            f.readline()
        for line in f:
            columns = line.split()
            columns = columns[0:2]
            line = ','.join(columns) + '\n'
            o.write(line)
