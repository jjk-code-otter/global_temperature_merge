from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

hadcrut5_dir = DATA_DIR / 'ManagedData' / 'Data' / 'HadCRUT5'
filename = hadcrut5_dir / 'HadCRUT.5.0.2.0.analysis.ensemble_series.global.annual.csv'

with open(hadcrut5_dir / 'ensemble_time_series.csv', 'w') as o:
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            columns = line.split(',')
            year = [columns[0]]
            ensemble = columns[3:]
            columns = year + ensemble
            line = ','.join(columns)
            o.write(line)
