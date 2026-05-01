from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'GISTEMP'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'GISTEMP'

data_file = pd.read_csv(data_file_dir / 'GLB.Ts+dSST.csv',skiprows=1)
data_file = data_file.apply(pd.to_numeric,errors='coerce').to_numpy()
data_file = data_file[~np.isnan(data_file[:,13]),:];
output = data_file[:,[0,13]]

np.savetxt(data_file_dir / "ensemble_time_series.csv", output, fmt='%.16f', delimiter=",")
