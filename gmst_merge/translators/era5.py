import numpy as np
import shutil
from gmst_merge.config import DATADIR, get_timestamp, quick_plot


def convert_file():
    timestamp = get_timestamp()
    data_file_dir = DATADIR / 'ERA5'
    filename = data_file_dir / 'C3S_Bulletin_temp_202512_Fig1b_timeseries_anomalies_ref1991-2020_global_allmonths_data.csv'
    ts_filename = data_file_dir / f'{timestamp}_C3S_Bulletin_temp_202512_Fig1b_timeseries_anomalies_ref1991-2020_global_allmonths_data.csv'

    shutil.copyfile(filename, ts_filename)

    years = []
    anoms = []
    with open(filename, 'r') as f:
        for i in range(12):
            f.readline()
        for line in f:
            columns = line.split(',')
            time = columns[0].split('-')
            years.append(int(time[0]))
            anoms.append(float(columns[3]))

    years = np.array(years)
    anoms = np.array(anoms)

    nyears = int(len(anoms) / 12)

    years = years[:nyears * 12]
    anoms = anoms[:nyears * 12]

    years = np.mean(years.reshape(nyears, 12), axis=1)
    anoms = np.mean(anoms.reshape(nyears, 12), axis=1)

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
    quick_plot('ERA5', ts_out_filename, f'../Figures/BasicInputPlots/{timestamp}_ERA5.png')

    output = np.zeros((nyears, 2))

    output[:, 0] = years[:]
    output[:, 1] = anoms[:] * 0.0 + 0.03

    out_filename = data_file_dir / "uncertainty_time_series.csv"
    ts_out_filename = data_file_dir / f"{timestamp}_uncertainty_time_series.csv"
    np.savetxt(
        out_filename,
        output,
        fmt='%.4f',
        delimiter=","
    )
    shutil.copy(out_filename, ts_out_filename)

if __name__ == '__main__':
    convert_file()
