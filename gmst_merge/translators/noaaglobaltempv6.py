from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

data_file_dir = os.getenv('DATADIR')
if data_file_dir is None:
    data_file_dir = Path(__file__).resolve().parent.parent / 'Data' / 'NOAAGlobalTempv6'
else:
    data_file_dir = data_file_dir / 'ManagedData' / 'Data' / 'NOAAGlobalTempv6'

# softcoded filename
matches = sorted(Path(data_file_dir).glob("aravg.ann.land_ocean.90S.90N.*.asc"))

file = open(matches[-1])
years = np.zeros((0,1))
mean = np.zeros((0,1))
while True:
    words = file.readline().split()
    if len(words) == 0:
        break
    years = np.append(years,float(words[0]))
    mean = np.append(mean,float(words[1]))
file.close()
years = years.reshape(-1,1)
mean = mean.reshape(-1,1)

np.savetxt(data_file_dir / "ensemble_time_series.csv", np.concatenate((years,mean),axis=1), fmt='%.16f', delimiter=",")
