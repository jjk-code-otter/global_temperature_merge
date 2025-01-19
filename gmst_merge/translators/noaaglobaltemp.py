from pathlib import Path
import xarray as xa
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

data_dir_env = os.getenv('DATADIR')
DATA_DIR = Path(data_dir_env)

hadcrut5_dir = DATA_DIR / 'ManagedData' / 'Data' / 'NOAA v6'
filename = hadcrut5_dir / 'aravg.ann.land_ocean.90S.90N.v6.0.0.202412.asc'

with open(hadcrut5_dir / 'ensemble_time_series.csv', 'w') as o:
    with open(filename, 'r') as f:
        for line in f:
            columns = line.split()
            columns = columns[0:2]
            line = ','.join(columns) + '\n'
            o.write(line)
