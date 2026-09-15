from pathlib import Path
import os
import datetime
import matplotlib.pyplot as plt
from gmst_merge.dataset import Dataset

DATADIR = os.getenv('DATADIR')
if DATADIR is None:
    DATADIR = Path(__file__).resolve().parent / 'Data'
else:
    DATADIR = Path(DATADIR) / 'ManagedData' / 'Data'

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def quick_plot(name, infilename, outfilename):

    ds = Dataset.read_csv_from_file(infilename, name)
    ds.anomalize(1981, 2010)

    plt.plot(ds.time[:], ds.data[:, 0])

    if ds.n_ensemble > 1:
        for i in range(1, ds.n_ensemble):
            plt.plot(ds.time[:], ds.data[:, i])

    plt.gca().set_ylim(-1.6, 1.6)
    plt.gca().set_xlim(1850, 2027)

    plt.savefig(outfilename)
    plt.close()