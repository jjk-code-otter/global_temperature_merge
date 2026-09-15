import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp, quick_plot


def convert_file():
    timestamp = get_timestamp()

    data_file_dir = DATADIR / 'CMA_GMST'
    filename = data_file_dir / 'CMA-GMST_Global_Month_Temp_1981_2010.csv'
    ts_filename = data_file_dir / f'{timestamp}_CMA-GMST_Global_Month_Temp_1981_2010.csv'

    shutil.copy(filename, ts_filename)

    years = []
    anoms = []
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            columns = line.split(',')
            years.append(int(columns[0]))

            monthly_anoms = columns[1:]
            monthly_anoms = [float(x) for x in monthly_anoms]
            monthly_anoms = np.array(monthly_anoms)

            anoms.append(np.mean(monthly_anoms))

    years = np.array(years)
    anoms = np.array(anoms)

    nyears = len(anoms)

    output = np.zeros((nyears, 2))

    output[:, 0] = years[:]
    output[:, 1] = anoms[:]

    out_filename = data_file_dir / "ensemble_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_ensemble_time_series.csv"
    np.savetxt(
        out_filename,
        output,
        fmt='%.4f',
        delimiter=","
    )
    shutil.copy(out_filename, ts_out_filename)
    quick_plot('CMA GMST', ts_out_filename, f'../Figures/BasicInputPlots/{timestamp}_CMA_GMST.png')


if __name__ == '__main__':
    convert_file()
