import numpy as np
import pandas as pd
import shutil
from gmst_merge.config import DATADIR, get_timestamp, quick_plot


def convert_file():
    timestamp = get_timestamp()
    data_file_dir = DATADIR / 'GISTEMP'
    filename = data_file_dir / 'GLB.Ts+dSST.csv'
    ts_filename = data_file_dir / f'{timestamp}_GLB.Ts+dSST.csv'

    shutil.copy(filename, ts_filename)

    data_file = pd.read_csv(filename, skiprows=1)
    data_file = data_file.apply(pd.to_numeric, errors='coerce').to_numpy()
    data_file = data_file[~np.isnan(data_file[:, 13]), :]
    output = data_file[:, [0, 13]]

    out_filename = data_file_dir / "ensemble_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_ensemble_time_series.csv"
    np.savetxt(
        out_filename,
        output,
        fmt='%.16f',
        delimiter=","
    )
    shutil.copy(out_filename, ts_out_filename)
    quick_plot('GISTEMP', ts_out_filename, f'../Figures/BasicInputPlots/{timestamp}_GISTEMP.png')


if __name__ == '__main__':
    convert_file()
